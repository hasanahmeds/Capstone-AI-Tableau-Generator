"""
Visualization Recommendation Engine
------------------------------------
Analyzes a DatasetSchema (and optionally a ProfileReport) to recommend
chart types that make sense for the data. Returns a DashboardSpec
ready for the Tableau generator to consume.

All models imported from schemas.py — VisualizationSpec, DashboardSpec,
DatasetSchema, ColumnSchema, ProfileReport, etc.

Supports 8 chart types: bar, line, scatter, histogram, box, heatmap,
pie, table.

Design choice: rule-based heuristics instead of ML because
  - Same input always gives same output (deterministic)
  - Every recommendation comes with a rationale string
  - No training data or extra dependencies needed
  - Easy to extend when we add new chart types later
"""

from typing import List, Optional, Dict, Any
from itertools import combinations

from scripts.schemas import (
    DatasetSchema,
    ColumnSchema,
    ProfileReport,
    ColumnProfile,
    NumericProfile,
    CategoricalProfile,
    VisualizationSpec,
    DashboardSpec,
)

from scripts.logger_config import get_logger

logger = get_logger("visualization_recommender")


# --- Thresholds ---------------------------------------------------------------
# Tuned by eyeballing a bunch of dashboards. If a categorical column has 200
# unique values, putting that on a bar chart is a mistake. These numbers
# keep the output reasonable.

MAX_PIE_CATEGORIES = 7       # pie gets unreadable if there are more than 7 slices
MAX_BAR_CATEGORIES = 10      # beyond this, bars become a color wall
MIN_ROWS_FOR_SCATTER = 30    # scatter needs enough points to mean anything; fewer → misleading
HIGH_CARDINALITY_LIMIT = 50  # above this, skip the column for grouping
MAX_VISUALS = 10             # cap so the dashboard doesn't become a wall of charts


class VisualizationRecommender:
    """
    Takes a DatasetSchema and produces a DashboardSpec with recommended
    visualizations.

    Usage:
        recommender = VisualizationRecommender()
        dashboard = recommender.recommend(dataset_schema)
        # or, if you have the profile report too:
        dashboard = recommender.recommend(dataset_schema, profile_report)

    The profile_report is optional — when provided, we use the numeric
    stats (min, max, mean, unique counts) to make smarter decisions about
    aggregations and chart types. Without it, we still work fine using
    just the column metadata from DatasetSchema.

    Rules fire in this order:
      1. Time series (line) — datetime + numeric
      2. Categorical breakdowns (bar, pie) — categorical + numeric
      3. Distributions (histogram, box) — numeric columns
      4. Relationships (scatter, heatmap) — pairs of numeric columns
      5. Summary table — fallback when nothing else fits
    """

    def __init__(self, max_visuals: int = MAX_VISUALS):
        self.max_visuals = max_visuals
        # stash profile data if provided, keyed by column name
        self._num_profiles: Dict[str, NumericProfile] = {}
        self._cat_profiles: Dict[str, CategoricalProfile] = {}
        self._col_schemas: Dict[str, ColumnSchema] = {}

    def recommend(
        self,
        ds: DatasetSchema,
        profile: Optional[ProfileReport] = None,
    ) -> DashboardSpec:
        """
        Main entry point. Analyzes the dataset schema and returns a
        DashboardSpec populated with VisualizationSpec objects.
        """
        visuals: List[VisualizationSpec] = []

        # index the column schemas by name for quick lookup
        self._col_schemas = {col.name: col for col in ds.columns}

        # if we got a profile report, index those too — the numeric stats
        # help us pick better aggregations and the unique counts help
        # decide whether pie charts are appropriate
        self._num_profiles = {}
        self._cat_profiles = {}
        if profile:
            for cp in profile.column_profiles:
                if cp.numeric:
                    self._num_profiles[cp.name] = cp.numeric
                if cp.categorical:
                    self._cat_profiles[cp.name] = cp.categorical

        logger.info(
            f"Recommending visuals for '{ds.name}': "
            f"{ds.row_count} rows, {ds.column_count} cols"
        )

        # Filter numeric columns to only meaningful measures.
        # Columns like "Row ID", "Postal Code", "Zip Code" are numeric
        # in pandas but are identifiers/codes, not measures to aggregate.
        # Summing or averaging them produces nonsense charts.
        _id_keywords = ("id", "code", "zip", "postal", "index", "number", "key", "fk", "pk")
        measure_cols = [
            c for c in ds.numeric_columns
            if not any(kw in c.lower().replace(" ", "").replace("_", "") for kw in _id_keywords)
        ]
        # Fallback: if ALL numeric cols got filtered out (unusual dataset), keep originals
        if not measure_cols:
            measure_cols = ds.numeric_columns

        logger.info(f"Measure columns (after filtering IDs): {measure_cols}")

        # --- Rule 1: time series ---
        if ds.datetime_columns and measure_cols:
            visuals.extend(
                self._rule_time_series(ds.datetime_columns, measure_cols, ds.categorical_columns)
            )

        # --- Rule 2: categorical breakdowns ---
        if ds.categorical_columns and measure_cols:
            visuals.extend(
                self._rule_categorical(ds.categorical_columns, measure_cols)
            )

        # --- Rule 3: distributions ---
        if measure_cols:
            visuals.extend(
                self._rule_distributions(measure_cols, ds.categorical_columns)
            )

        # --- Rule 4: numeric relationships ---
        # scatter plots for pairs of meaningful measure columns
        if len(measure_cols) >= 2 and ds.row_count >= MIN_ROWS_FOR_SCATTER:
            visuals.extend(
                self._rule_numeric_relationships(measure_cols)
            )

        # --- Rule 5: summary table as a fallback ---
        # if we haven't generated much, add a table so the dashboard
        # isn't empty
        if len(visuals) < 2:
            visuals.extend(self._rule_table_fallback(ds))

        # trim if we went overboard — sort by confidence descending,
        # keep the top N so the dashboard stays digestible
        if len(visuals) > self.max_visuals:
            visuals.sort(key=lambda v: (v.confidence or 0.0), reverse=True)
            visuals = visuals[:self.max_visuals]

        # build the layout grid — 2 columns, charts flow left to right
        layout = {
            "type": "grid",
            "columns": 2,
            "order": list(range(len(visuals))),
        }

        dashboard = DashboardSpec(
            name=f"{ds.name} Dashboard",
            visuals=visuals,
            layout=layout,
        )

        logger.info(f"Generated {len(visuals)} visualizations.")
        return dashboard

    # RULE 1 — TIME SERIES (LINE)

    def _rule_time_series(
        self,
        dt_cols: List[str],
        num_cols: List[str],
        cat_cols: List[str],
    ) -> List[VisualizationSpec]:
        """
        Datetime + numeric -> line chart. We pair the first datetime column
        with up to 3 numeric columns. If there's a low-cardinality categorical
        column, we use it as a group_by so the user gets separate lines
        (ex: revenue over time by region).
        """
        specs = []
        time_col = dt_cols[0]
        group_col = self._pick_grouping_column(cat_cols)

        for num_col in num_cols[:3]:
            agg = self._guess_aggregation(num_col)
            spec = VisualizationSpec(
                chart_type="line",
                title=f"{num_col} over time",
                x=time_col,
                y=num_col,
                aggregation=agg,
                group_by=[group_col] if group_col else None,
                rationale=(
                    f"Time trend for '{num_col}' across '{time_col}'. "
                    f"Line charts are the go-to for temporal patterns."
                    + (f" Grouped by '{group_col}' for comparison." if group_col else "")
                ),
                confidence=0.85,
            )
            specs.append(spec)

        return specs

    # RULE 2 — CATEGORICAL BREAKDOWNS (BAR, PIE)

    def _rule_categorical(
        self,
        cat_cols: List[str],
        num_cols: List[str],
    ) -> List[VisualizationSpec]:
        """
        Category + numeric -> bar chart. If the category has very few
        unique values (<=7), we also suggest a pie chart for the
        part-to-whole view.

        We skip columns with too many unique values because bars become
        unreadable at that point.
        """
        specs = []
        primary_num = num_cols[0]
        agg = self._guess_aggregation(primary_num)

        # Sort categorical columns by cardinality (low first) so we pick
        # useful columns like "Segment", "Region", "Category" instead of
        # high-cardinality ones like "Order ID" or "Customer Name"
        scored_cats = []
        for cat_col in cat_cols:
            uc = self._get_unique_count(cat_col)
            if uc is not None and (uc < 2 or uc > HIGH_CARDINALITY_LIMIT):
                continue  # skip single-value and high-cardinality columns
            scored_cats.append((uc if uc is not None else 999, cat_col))
        scored_cats.sort(key=lambda t: t[0])
        best_cats = [col for _, col in scored_cats[:3]]

        for cat_col in best_cats:
            unique_count = self._get_unique_count(cat_col)

            # bar chart — almost always a safe pick
            bar = VisualizationSpec(
                chart_type="bar",
                title=f"{primary_num} by {cat_col}",
                x=cat_col,
                y=primary_num,
                aggregation=agg,
                rationale=(
                    f"Compare '{primary_num}' across '{cat_col}' categories. "
                    f"Bar charts handle this well"
                    + (f" ({unique_count} categories)." if unique_count else ".")
                ),
                confidence=0.80,
            )
            specs.append(bar)

            # pie chart only when cardinality is low enough to be readable
            effective_uniques = unique_count if unique_count else 0
            if 2 <= effective_uniques <= MAX_PIE_CATEGORIES:
                pie = VisualizationSpec(
                    chart_type="pie",
                    title=f"{primary_num} share by {cat_col}",
                    x=cat_col,
                    y=primary_num,
                    aggregation=agg,
                    rationale=(
                        f"Part-to-whole breakdown with only {effective_uniques} "
                        f"categories — pie chart is readable here."
                    ),
                    confidence=0.65,
                )
                specs.append(pie)

        return specs

    # RULE 3 — DISTRIBUTIONS (HISTOGRAM, BOX)

    def _rule_distributions(
        self,
        num_cols: List[str],
        cat_cols: List[str],
    ) -> List[VisualizationSpec]:
        """
        Histogram for individual numeric columns to show how values
        are spread out. Box plots when we have a categorical column
        to split by — lets you compare distributions across groups.
        """
        specs = []

        # histograms for up to 2 numeric columns
        for col in num_cols[:2]:
            hist = VisualizationSpec(
                chart_type="histogram",
                title=f"Distribution of {col}",
                x=col,
                rationale=(
                    f"Shows frequency distribution for '{col}'. "
                    f"Good for spotting skewness, outliers, or multiple modes."
                ),
                confidence=0.70,
            )
            specs.append(hist)

        # box plot: numeric grouped by a categorical dimension
        group_col = self._pick_grouping_column(cat_cols)
        if num_cols and group_col:
            box = VisualizationSpec(
                chart_type="box",
                title=f"{num_cols[0]} by {group_col}",
                x=group_col,
                y=num_cols[0],
                group_by=[group_col],
                rationale=(
                    f"Compare spread and outliers of '{num_cols[0]}' "
                    f"across '{group_col}' groups."
                ),
                confidence=0.65,
            )
            specs.append(box)

        return specs

    # RULE 4 — NUMERIC RELATIONSHIPS (SCATTER, HEATMAP)

    def _rule_numeric_relationships(
        self,
        num_cols: List[str],
    ) -> List[VisualizationSpec]:
        """
        Two numeric columns -> scatter plot to look for correlations.
        Three or more numeric columns -> also add a correlation heatmap,
        which gives a quick birds-eye view of what moves together.

        We cap scatter plots at 3 pairs to keep the dashboard manageable.
        """
        specs = []

        # scatter for up to 3 column pairs
        pairs = list(combinations(num_cols, 2))[:3]
        for col_x, col_y in pairs:
            scatter = VisualizationSpec(
                chart_type="scatter",
                title=f"{col_y} vs {col_x}",
                x=col_x,
                y=col_y,
                rationale=(
                    f"Scatter plot to check for correlation between "
                    f"'{col_x}' and '{col_y}'."
                ),
                confidence=0.68,
            )
            specs.append(scatter)

        # Note: correlation heatmap removed — Tableau doesn't have a built-in
        # correlation matrix chart type, and generating one requires calculated
        # fields for each pair which is beyond what the XML generator supports.

        return specs

    # RULE 5 — TABLE FALLBACK

    def _rule_table_fallback(self, ds: DatasetSchema) -> List[VisualizationSpec]:
        """
        If the other rules didn't produce much, we add a summary table
        so the dashboard isn't empty. This can happen with very sparse
        or unusual datasets.
        """
        # pick up to 5 columns for the table — mix of types
        table_cols = []
        for source in [ds.categorical_columns, ds.numeric_columns,
                       ds.datetime_columns, ds.text_columns]:
            for col in source:
                if col not in table_cols:
                    table_cols.append(col)
                if len(table_cols) >= 5:
                    break
            if len(table_cols) >= 5:
                break

        if not table_cols:
            return []

        return [
            VisualizationSpec(
                chart_type="table",
                title="Data summary",
                x=table_cols[0],
                group_by=table_cols,
                rationale=(
                    "Tabular overview of key columns. Added because the "
                    "dataset didn't trigger enough chart-specific rules."
                ),
                confidence=0.50,
            )
        ]

    # HELPER METHODS

    def _get_unique_count(self, col_name: str) -> Optional[int]:
        """
        Try to get the unique count for a column. First check the
        profile report (CategoricalProfile.unique), then go back to
        ColumnSchema.unique_count. Returns None if we have no unique count.
        """
        if col_name in self._cat_profiles:
            return self._cat_profiles[col_name].unique
        schema = self._col_schemas.get(col_name)
        if schema and schema.unique_count is not None:
            return schema.unique_count
        return None

    def _guess_aggregation(self, col_name: str) -> str:
        """
        Pick an aggregation based on the column name and stats.

        Columns whose names contain 'rate', 'score', 'avg', 'pct' etc.
        probably make more sense with 'avg'. Columns with 'count' or
        'qty' should be summed. Values in the 0-1 range are likely
        proportions so we average those too.

        Default is 'sum' because that's what most people expect.
        """
        name = col_name.lower()

        # names that suggest averaging
        avg_hints = ["rate", "ratio", "pct", "percentage", "avg", "average",
                     "score", "rating", "index", "per_"]
        if any(h in name for h in avg_hints):
            return "avg"

        # names that suggest summing
        sum_hints = ["count", "num_", "number_of", "qty", "quantity", "total"]
        if any(h in name for h in sum_hints):
            return "sum"

        # if the profile says values are in [0, 1], it's probably a proportion/rate
        np = self._num_profiles.get(col_name)
        if np and np.min is not None and np.max is not None:
            if np.min >= 0 and np.max <= 1:
                return "avg"

        return "sum"

    def _pick_grouping_column(self, cat_cols: List[str]) -> Optional[str]:
        """
        Find the best categorical column for color-coding or grouping.
        Sweet spot is 3-10 unique values — enough to be interesting
        but few enough that the legend stays clean.

        If we don't have unique counts, we just take the first
        categorical column and hope for the best.
        """
        scored = []
        for col in cat_cols:
            uniques = self._get_unique_count(col)
            if uniques is not None:
                # skip anything with only 1 value or way too many
                if uniques < 2 or uniques > 12:
                    continue
                # prefer things close to 5 categories
                distance = abs(uniques - 5)
                scored.append((distance, col))
            else:
                # no unique-count info available — assign a middling penalty score
                # so it might still be picked if nothing better exists
                scored.append((10, col))

        if not scored:
            return None

        scored.sort(key=lambda t: t[0])
        return scored[0][1]
#-------------------------------------------------------------
# the main function is to check if the module working properly
# python visualization_recommender.py

if __name__ == "__main__":
    #imported required modules for main function few blocks are repeated again to check this modules
    import json
    import os
    import pandas as pd
    from scripts.schemas import detect_semantic_type, DatasetSchema, ColumnSchema
    from scripts.schemas import build_profile_report

    # ---- swap this to your CSV path ----
    FILE_PATH = "train.csv"

    df = pd.read_csv(FILE_PATH)

    print(f"Loaded: {FILE_PATH}")
    print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")
    print(f"  Columns: {list(df.columns)}\n")

    # --- build DatasetSchema using the detect_semantic_type helper ---
    col_schemas = []
    numeric_cols, cat_cols, dt_cols = [], [], []
    bool_cols, text_cols, unknown_cols = [], [], []

    for col_name in df.columns:
        sem_type, meta = detect_semantic_type(df[col_name])
        cs = ColumnSchema(
            name=col_name,
            dtype_raw=str(df[col_name].dtype),
            semantic_type=sem_type,
            missing_count=int(df[col_name].isna().sum()),
            missing_ratio=float(df[col_name].isna().mean()),
            unique_count=int(df[col_name].nunique()),
            sample_values=df[col_name].dropna().head(5).tolist(),
        )
        col_schemas.append(cs)

        # bucket into the right list
        if sem_type == "numeric":
            numeric_cols.append(col_name)
        elif sem_type == "categorical":
            cat_cols.append(col_name)
        elif sem_type == "datetime":
            dt_cols.append(col_name)
        elif sem_type == "boolean":
            bool_cols.append(col_name)
        elif sem_type == "text":
            text_cols.append(col_name)
        else:
            unknown_cols.append(col_name)

    ds = DatasetSchema(
        name=FILE_PATH,
        row_count=len(df),
        column_count=len(df.columns),
        columns=col_schemas,
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
        datetime_columns=dt_cols,
        boolean_columns=bool_cols,
        text_columns=text_cols,
        unknown_columns=unknown_cols,
    )

    # --- build profile report for richer recommendations ---
    profile = build_profile_report(ds, df)

    # --- run the recommender ---
    recommender = VisualizationRecommender()
    dashboard = recommender.recommend(ds, profile)

    # --- print results ---
    print(f"{'=' * 55}")
    print(f"  Dashboard: {dashboard.name}")
    print(f"  {len(dashboard.visuals)} Visualizations")
    print(f"{'=' * 55}\n")

    for i, viz in enumerate(dashboard.visuals, 1):
        print(f"  {i}. [{viz.chart_type.upper()}] {viz.title}")
        if viz.x:
            print(f"     X: {viz.x}", end="")
            if viz.y:
                print(f"  |  Y: {viz.y}", end="")
            print()
        if viz.aggregation:
            print(f"     Aggregation: {viz.aggregation}")
        if viz.group_by:
            print(f"     Group by: {viz.group_by}")
        if viz.confidence is not None:
            print(f"     Confidence: {viz.confidence:.0%}")
        if viz.rationale:
            print(f"     Rationale: {viz.rationale}")
        print()

    # save results to JSON for future use
    output = dashboard.model_dump()
    os.makedirs("logs", exist_ok=True)
    with open("logs/recommendations_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results saved to logs/recommendations_output.json")