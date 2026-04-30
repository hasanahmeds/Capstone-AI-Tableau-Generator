import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Literal, Tuple
from uuid import uuid4

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict, field_validator


# ─────────────────────────────────────────────────────────────
# Type Aliases
# ─────────────────────────────────────────────────────────────

ColumnSemanticType = Literal["numeric", "categorical", "datetime", "boolean", "text", "unknown"]
FileFormat = Literal["csv", "excel"]
Severity = Literal["low", "medium", "high", "critical"]
ChartType = Literal["bar", "line", "scatter", "histogram", "box", "heatmap", "pie", "treemap", "table"]


# ─────────────────────────────────────────────────────────────
# Column & Dataset Schemas
# ─────────────────────────────────────────────────────────────

class ColumnSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    dtype_raw: Optional[str] = None
    semantic_type: ColumnSemanticType = "unknown"

    missing_count: int = 0
    missing_ratio: float = 0.0
    unique_count: Optional[int] = None
    is_constant: bool = False
    is_empty: bool = False
    sample_values: Optional[List[Any]] = None


class DatasetSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = "uploaded_dataset"
    file_format: FileFormat = "csv"
    source_filename: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    row_count: int = 0
    column_count: int = 0
    columns: List[ColumnSchema] = Field(default_factory=list)

    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    boolean_columns: List[str] = Field(default_factory=list)
    text_columns: List[str] = Field(default_factory=list)
    unknown_columns: List[str] = Field(default_factory=list)

    preview_rows: Optional[List[Dict[str, Any]]] = None


# ─────────────────────────────────────────────────────────────
# Quality Report Schemas
# ─────────────────────────────────────────────────────────────

class TypeIssue(BaseModel):
    model_config = ConfigDict(extra="allow")

    column: str
    issue: str
    count: Optional[int] = None
    examples: Optional[List[Any]] = None
    severity: Severity = "medium"


class MissingValueReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    by_column: Dict[str, int] = Field(default_factory=dict)
    total_missing: int = 0
    missing_ratio_overall: float = 0.0  # 0..1


class DuplicateReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    duplicate_row_count: int = 0
    duplicate_row_ratio: float = 0.0  # 0..1


class OutlierReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    method: Literal["iqr"] = "iqr"
    by_column: Dict[str, int] = Field(default_factory=dict)
    bounds_by_column: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class QualityReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    missing: MissingValueReport = Field(default_factory=MissingValueReport)
    duplicates: DuplicateReport = Field(default_factory=DuplicateReport)
    outliers: Optional[OutlierReport] = None

    empty_columns: List[str] = Field(default_factory=list)
    constant_columns: List[str] = Field(default_factory=list)
    type_issues: List[TypeIssue] = Field(default_factory=list)

    quality_score: Optional[float] = None
    score_scale: Literal["0_100"] = "0_100"

    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)

    @field_validator("quality_score")
    @classmethod
    def score_non_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if v < 0:
            raise ValueError("quality_score cannot be negative.")
        return v


# ─────────────────────────────────────────────────────────────
# Profile Report Schemas
# ─────────────────────────────────────────────────────────────

class NumericProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    count: int
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    p25: Optional[float] = None
    median: Optional[float] = None
    p75: Optional[float] = None
    max: Optional[float] = None


class CategoricalProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    count: int
    unique: int
    top: Optional[Any] = None
    freq: Optional[int] = None
    top_k: Optional[List[Dict[str, Any]]] = None  # [{"value":..., "count":...}, ...]


class DatetimeProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    count: int
    min: Optional[str] = None  # ISO string
    max: Optional[str] = None


class TextProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    count: int
    avg_length: Optional[float] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None


class ColumnProfile(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    semantic_type: ColumnSemanticType = "unknown"
    missing_count: int = 0
    missing_ratio: float = 0.0

    numeric: Optional[NumericProfile] = None
    categorical: Optional[CategoricalProfile] = None
    datetime: Optional[DatetimeProfile] = None
    text: Optional[TextProfile] = None


class ProfileReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    column_profiles: List[ColumnProfile] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────
# Visualization & Dashboard Schemas
# ─────────────────────────────────────────────────────────────

class VisualizationSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    chart_type: ChartType
    title: str
    x: Optional[str] = None
    y: Optional[str] = None
    aggregation: Optional[Literal["sum", "avg", "min", "max", "count", "median"]] = None
    group_by: Optional[List[str]] = None
    rationale: Optional[str] = None
    confidence: Optional[float] = None

    @field_validator("confidence")
    @classmethod
    def conf_between_0_1(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if not (0.0 <= v <= 1.0):
            raise ValueError("confidence must be between 0 and 1.")
        return v


class DashboardSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "Auto Generated Dashboard"
    visuals: List[VisualizationSpec] = Field(default_factory=list)
    layout: Optional[Dict[str, Any]] = None
    global_filters: Optional[List[Dict[str, Any]]] = None


# ─────────────────────────────────────────────────────────────
# Type Detection Helpers
# ─────────────────────────────────────────────────────────────

BOOL_TRUE = {"true", "t", "yes", "y", "1"}
BOOL_FALSE = {"false", "f", "no", "n", "0"}


def detect_semantic_type(
    series: pd.Series,
    datetime_threshold: float = 0.8,
) -> Tuple[ColumnSemanticType, Dict[str, Any]]:
    """
    Detect the semantic type of a pandas Series.

    Returns:
        (semantic_type, extra_info) where extra_info may contain
        parsed_datetime_success_ratio, bool_mapping, unique_count, etc.
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return "unknown", {"reason": "empty_or_all_null"}

    # Numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric", {}

    # Boolean (common string/0/1 patterns)
    vals = non_null.astype(str).str.strip().str.lower()
    unique_vals = set(vals.unique())
    if unique_vals.issubset(BOOL_TRUE.union(BOOL_FALSE)):
        return "boolean", {"unique_vals": sorted(list(unique_vals))}

    # Datetime
    parsed = pd.to_datetime(non_null, errors="coerce")
    success_ratio = float(parsed.notna().mean()) if len(parsed) else 0.0
    if success_ratio >= datetime_threshold:
        return "datetime", {"parsed_success_ratio": success_ratio}

    # Text vs Categorical
    unique_count = int(non_null.nunique())
    avg_len = float(non_null.astype(str).str.len().mean())
    if unique_count > 50 or avg_len > 30:
        return "text", {"unique_count": unique_count, "avg_len": avg_len}

    return "categorical", {"unique_count": unique_count, "avg_len": avg_len}


def iqr_outlier_count(
    series: pd.Series,
    iqr_multiplier: float = 1.5,
) -> Tuple[int, Optional[float], Optional[float]]:
    """
    Count outliers using the IQR method.

    Returns:
        (outlier_count, lower_bound, upper_bound)
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4:
        return 0, None, None

    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr == 0:
        return 0, None, None

    low = q1 - iqr_multiplier * iqr
    high = q3 + iqr_multiplier * iqr
    outliers = int(((s < low) | (s > high)).sum())
    return outliers, low, high


# ─────────────────────────────────────────────────────────────
# Builder Functions
# ─────────────────────────────────────────────────────────────

def build_dataset_schema(
    file_name: str,
    df: pd.DataFrame,
    preview_n: int = 10,
) -> DatasetSchema:
    """Build a DatasetSchema from a DataFrame."""
    dataset_id = str(uuid4())

    col_schemas: List[ColumnSchema] = []
    numeric_cols, cat_cols, dt_cols, bool_cols, text_cols, unknown_cols = (
        [], [], [], [], [], [],
    )

    for col in df.columns:
        s = df[col]
        dtype_raw = str(s.dtype)

        semantic, extra = detect_semantic_type(s)

        missing_count = int(s.isna().sum())
        missing_ratio = float(missing_count / len(df)) if len(df) else 0.0

        non_null = s.dropna()
        unique_count = int(non_null.nunique()) if len(non_null) else 0
        is_empty = len(non_null) == 0
        is_constant = unique_count == 1 and not is_empty

        sample_values = non_null.head(5).tolist() if len(non_null) else []

        # Remove 'unique_count' from extra if present (passed explicitly)
        extra.pop("unique_count", None)

        cs = ColumnSchema(
            name=col,
            dtype_raw=dtype_raw,
            semantic_type=semantic,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
            unique_count=unique_count,
            is_empty=is_empty,
            is_constant=is_constant,
            sample_values=sample_values,
            **extra,
        )
        col_schemas.append(cs)

        if semantic == "numeric":
            numeric_cols.append(col)
        elif semantic == "categorical":
            cat_cols.append(col)
        elif semantic == "datetime":
            dt_cols.append(col)
        elif semantic == "boolean":
            bool_cols.append(col)
        elif semantic == "text":
            text_cols.append(col)
        else:
            unknown_cols.append(col)

    preview_rows = df.head(preview_n).to_dict(orient="records")

    return DatasetSchema(
        dataset_id=dataset_id,
        name=file_name,
        file_format="csv",
        source_filename=file_name,
        row_count=int(len(df)),
        column_count=int(len(df.columns)),
        columns=col_schemas,
        numeric_columns=numeric_cols,
        categorical_columns=cat_cols,
        datetime_columns=dt_cols,
        boolean_columns=bool_cols,
        text_columns=text_cols,
        unknown_columns=unknown_cols,
        preview_rows=preview_rows,
    )


def build_quality_report(
    ds: DatasetSchema,
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5,
) -> QualityReport:
    """Build a QualityReport from a DatasetSchema and DataFrame."""

    # Missing
    missing_by_col = {c: int(df[c].isna().sum()) for c in df.columns}
    total_missing = int(sum(missing_by_col.values()))
    missing_ratio_overall = (
        float(total_missing / (len(df) * len(df.columns)))
        if len(df) and len(df.columns)
        else 0.0
    )

    missing_report = MissingValueReport(
        by_column=missing_by_col,
        total_missing=total_missing,
        missing_ratio_overall=missing_ratio_overall,
    )

    # Duplicates
    dup_count = int(df.duplicated().sum())
    dup_ratio = float(dup_count / len(df)) if len(df) else 0.0
    dup_report = DuplicateReport(
        duplicate_row_count=dup_count,
        duplicate_row_ratio=dup_ratio,
    )

    # Empty / Constant columns
    empty_cols = [c.name for c in ds.columns if c.is_empty]
    constant_cols = [c.name for c in ds.columns if c.is_constant]

    # Type issues
    type_issues: List[TypeIssue] = []

    for c in ds.numeric_columns:
        s = df[c]
        coerced = pd.to_numeric(s, errors="coerce")
        failures = int(coerced.isna().sum() - s.isna().sum())
        if failures > 0:
            examples = s[coerced.isna() & s.notna()].astype(str).head(5).tolist()
            type_issues.append(
                TypeIssue(
                    column=c,
                    issue="numeric_coercion_failed",
                    count=failures,
                    examples=examples,
                    severity="high",
                )
            )

    for c in ds.datetime_columns:
        s = df[c]
        parsed = pd.to_datetime(s, errors="coerce")
        failures = int(parsed.isna().sum() - s.isna().sum())
        if failures > 0:
            examples = s[parsed.isna() & s.notna()].astype(str).head(5).tolist()
            type_issues.append(
                TypeIssue(
                    column=c,
                    issue="datetime_parse_failed",
                    count=failures,
                    examples=examples,
                    severity="high",
                )
            )

    # Outliers (IQR)
    out_by_col: Dict[str, int] = {}
    bounds_by_col: Dict[str, Dict[str, float]] = {}

    for c in ds.numeric_columns:
        out_cnt, low, high = iqr_outlier_count(df[c], iqr_multiplier=iqr_multiplier)
        out_by_col[c] = out_cnt
        if low is not None and high is not None:
            bounds_by_col[c] = {"low": float(low), "high": float(high)}

    out_report = (
        OutlierReport(method="iqr", by_column=out_by_col, bounds_by_column=bounds_by_col)
        if ds.numeric_columns
        else None
    )

    # Quality score (0..100)
    penalties = 0.0
    penalties += missing_ratio_overall * 60.0
    penalties += dup_ratio * 20.0
    penalties += (len(empty_cols) / max(1, len(df.columns))) * 10.0
    penalties += (len(constant_cols) / max(1, len(df.columns))) * 5.0

    if ds.numeric_columns and len(df) > 0:
        outlier_ratio_avg = sum(out_by_col.values()) / (
            len(ds.numeric_columns) * len(df)
        )
        penalties += outlier_ratio_avg * 20.0

    quality_score = max(0.0, 100.0 - penalties)

    warnings: List[str] = []
    if empty_cols:
        warnings.append(f"Found empty columns: {empty_cols}")
    if constant_cols:
        warnings.append(f"Found constant columns: {constant_cols}")
    if dup_count > 0:
        warnings.append(f"Found {dup_count} duplicate rows.")
    if total_missing > 0:
        warnings.append(f"Found {total_missing} missing values in total.")

    return QualityReport(
        dataset_id=ds.dataset_id,
        missing=missing_report,
        duplicates=dup_report,
        outliers=out_report,
        empty_columns=empty_cols,
        constant_columns=constant_cols,
        type_issues=type_issues,
        quality_score=round(quality_score, 2),
        warnings=warnings,
        errors=[],
    )


def build_profile_report(
    ds: DatasetSchema,
    df: pd.DataFrame,
    top_k: int = 5,
) -> ProfileReport:
    """Build a ProfileReport from a DatasetSchema and DataFrame."""
    profiles: List[ColumnProfile] = []
    notes: List[str] = []

    for col_meta in ds.columns:
        c = col_meta.name
        s = df[c]
        missing_count = int(s.isna().sum())
        missing_ratio = float(missing_count / len(df)) if len(df) else 0.0
        semantic = col_meta.semantic_type

        cp = ColumnProfile(
            name=c,
            semantic_type=semantic,
            missing_count=missing_count,
            missing_ratio=missing_ratio,
        )

        if semantic == "numeric":
            numeric = pd.to_numeric(s, errors="coerce").dropna()
            if len(numeric) > 0:
                cp.numeric = NumericProfile(
                    count=int(len(numeric)),
                    mean=float(numeric.mean()),
                    std=float(numeric.std(ddof=1)) if len(numeric) > 1 else 0.0,
                    min=float(numeric.min()),
                    p25=float(numeric.quantile(0.25)),
                    median=float(numeric.quantile(0.50)),
                    p75=float(numeric.quantile(0.75)),
                    max=float(numeric.max()),
                )
            else:
                notes.append(
                    f"Numeric column '{c}' has no valid numeric values after coercion."
                )

        elif semantic in ("categorical", "boolean"):
            non_null = s.dropna()
            uniq = int(non_null.nunique()) if len(non_null) else 0
            vc = non_null.astype(str).value_counts(dropna=True)
            top_val = vc.index[0] if len(vc) else None
            top_freq = int(vc.iloc[0]) if len(vc) else None
            top_list = (
                [{"value": k, "count": int(v)} for k, v in vc.head(top_k).items()]
                if len(vc)
                else []
            )
            cp.categorical = CategoricalProfile(
                count=int(len(non_null)),
                unique=uniq,
                top=top_val,
                freq=top_freq,
                top_k=top_list,
            )

        elif semantic == "datetime":
            parsed = pd.to_datetime(
                s, errors="coerce"
            ).dropna()
            if len(parsed) > 0:
                cp.datetime = DatetimeProfile(
                    count=int(len(parsed)),
                    min=str(parsed.min().to_pydatetime().isoformat()),
                    max=str(parsed.max().to_pydatetime().isoformat()),
                )
            else:
                notes.append(
                    f"Datetime column '{c}' could not be parsed to datetime."
                )

        elif semantic == "text":
            non_null = s.dropna().astype(str)
            if len(non_null) > 0:
                lengths = non_null.str.len()
                cp.text = TextProfile(
                    count=int(len(non_null)),
                    avg_length=float(lengths.mean()),
                    min_length=int(lengths.min()),
                    max_length=int(lengths.max()),
                )

        profiles.append(cp)

    return ProfileReport(
        dataset_id=ds.dataset_id,
        column_profiles=profiles,
        notes=notes,
    )


def build_dashboard_spec(
    ds: DatasetSchema,
    df: pd.DataFrame,
) -> DashboardSpec:
    """Build a basic DashboardSpec with auto-recommended visualizations.

    Uses proposal heuristics:
      • Time + metric → Line chart
      • 1 numeric → Histogram
      • 2 continuous → Scatter
      • 1 category + 1 metric → Bar chart
    Skips identifier columns (columns with 'id', 'code', 'key' etc. in their name)
    """
    visuals: List[VisualizationSpec] = []

    # Identify real measures (skip identifiers like Row ID, Postal Code)
    _id_tokens = {"id", "uuid", "guid", "code", "key", "postal", "zip", "row"}
    def _is_id(name):
        tokens = {t.strip().lower() for t in name.replace("_", " ").split()}
        return bool(tokens & _id_tokens)

    real_numeric = [c for c in ds.numeric_columns if not _is_id(c)]
    real_categorical = [c for c in ds.categorical_columns if not _is_id(c)]

    # 1) Datetime + numeric → line chart
    if ds.datetime_columns and real_numeric:
        visuals.append(
            VisualizationSpec(
                chart_type="line",
                title=f"{real_numeric[0]} over time",
                x=ds.datetime_columns[0],
                y=real_numeric[0],
                aggregation="sum",
                rationale="Time + metric → Line chart.",
                confidence=0.8,
            )
        )

    # 2) At least 1 numeric → histogram
    if real_numeric:
        visuals.append(
            VisualizationSpec(
                chart_type="histogram",
                title=f"Distribution of {real_numeric[0]}",
                x=real_numeric[0],
                rationale="1 metric → Distribution chart.",
                confidence=0.75,
            )
        )

    # 3) At least 2 numeric → scatter
    if len(real_numeric) >= 2:
        visuals.append(
            VisualizationSpec(
                chart_type="scatter",
                title=f"{real_numeric[0]} vs {real_numeric[1]}",
                x=real_numeric[0],
                y=real_numeric[1],
                rationale="2 continuous → Scatter plot.",
                confidence=0.7,
            )
        )

    # 4) Categorical + numeric → chart type by cardinality
    if real_categorical and real_numeric:
        cat = real_categorical[0]
        n_unique = df[cat].nunique() if cat in df.columns else 0
        if n_unique <= 6:
            ct, rat = "pie", "1 category (few values) + 1 metric → Pie chart."
        elif n_unique <= 25:
            ct, rat = "bar", "1 category + 1 metric → Bar chart."
        else:
            ct, rat = "treemap", "Hierarchical breakdown → Treemap."
        visuals.append(
            VisualizationSpec(
                chart_type=ct,
                title=f"{real_numeric[0]} by {cat}",
                x=cat,
                y=real_numeric[0],
                aggregation="sum",
                rationale=rat,
                confidence=0.72,
            )
        )

    return DashboardSpec(
        name="Auto Generated Dashboard",
        visuals=visuals,
        layout={"type": "grid", "columns": 2, "order": list(range(len(visuals)))},
    )


# ─────────────────────────────────────────────────────────────
# Workbook Generator Internal Schemas
# ─────────────────────────────────────────────────────────────

class WorksheetSpec(BaseModel):
    """
    Concrete Tableau worksheet plan produced after converting a
    VisualizationSpec into Tableau shelf assignments.

    Used in TableauWorkbookGenerator._plan_worksheets()
    and later in XML generation.
    """

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    name: str = Field(..., description="Worksheet name in Tableau")
    mark_type: Literal["Bar", "Line", "Circle", "Square", "Pie", "Text"] = Field(
        ...,
        description="Tableau mark type"
    )

    rows_shelf: List[str] = Field(
        default_factory=list,
        description="Fields placed on Rows shelf"
    )
    columns_shelf: List[str] = Field(
        default_factory=list,
        description="Fields placed on Columns shelf"
    )

    color_field: Optional[str] = Field(
        default=None,
        description="Field used for color encoding"
    )
    size_field: Optional[str] = Field(
        default=None,
        description="Field used for size encoding"
    )
    tooltip_fields: List[str] = Field(
        default_factory=list,
        description="Fields shown in tooltip"
    )

    aggregation: Optional[Literal["SUM", "AVG", "MIN", "MAX", "COUNT", "MEDIAN"]] = Field(
        default=None,
        description="Tableau aggregation used for measures"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v:
            raise ValueError("Worksheet name cannot be empty")
        return v

    @field_validator("rows_shelf", "columns_shelf", "tooltip_fields")
    @classmethod
    def clean_field_lists(cls, v: List[str]) -> List[str]:
        return [item.strip() for item in v if item and item.strip()]


class WorkbookSpec(BaseModel):
    """
    Internal workbook plan used by TableauWorkbookGenerator after
    worksheet planning is complete.

    This is the object returned by _plan_worksheets().
    """

    model_config = ConfigDict(
        extra="allow",
        str_strip_whitespace=True,
    )

    name: str = Field(..., description="Workbook/dashboard title")
    datasource_name: str = Field(
        ...,
        description="Internal Tableau datasource name"
    )
    worksheets: List[WorksheetSpec] = Field(
        default_factory=list,
        description="All worksheets in the workbook"
    )
    dashboard_width: int = Field(
        default=1200,
        ge=100,
        description="Dashboard canvas width in pixels"
    )
    dashboard_height: int = Field(
        default=900,
        ge=100,
        description="Dashboard canvas height in pixels"
    )

    @field_validator("name", "datasource_name")
    @classmethod
    def validate_required_strings(cls, v: str) -> str:
        if not v:
            raise ValueError("This field cannot be empty")
        return v


# ─────────────────────────────────────────────────────────────
# Workflow State Schemas (used by DashboardGeneratorWorkflow)
# ─────────────────────────────────────────────────────────────

# Every stage that the pipeline can be in, ordered from start to finish.
WorkflowStage = Literal[
    "pending",
    "validating",
    "profiling",
    "analyzing",
    "recommending",
    "generating",
    "finalizing",
    "completed",
    "failed",
]


class KPIRecommendation(BaseModel):
    """Single KPI suggested by the AI analysis engine."""
    model_config = ConfigDict(extra="allow")

    metric_name: str = Field(..., description="Human-readable KPI name")
    column: str = Field(..., description="Source column in the dataset")
    aggregation: Literal["sum", "avg", "min", "max", "count", "median"] = "sum"
    rationale: Optional[str] = None


class AnalysisResult(BaseModel):
    """
    Captures everything returned by the AI analysis engine:
    recommended KPIs, a short summary of dataset characteristics,
    and any additional metadata the LLM provided.
    """
    model_config = ConfigDict(extra="allow")

    dataset_id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    summary: str = ""
    kpis: List[KPIRecommendation] = Field(default_factory=list)
    suggested_chart_types: List[ChartType] = Field(default_factory=list)
    business_domain: Optional[str] = None
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowError(BaseModel):
    """Structured record of an error that happened during a workflow step."""
    model_config = ConfigDict(extra="allow")

    stage: WorkflowStage
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    recoverable: bool = True
    details: Optional[Dict[str, Any]] = None


class WorkflowProgress(BaseModel):
    """Lightweight progress tracker shown in the UI."""
    model_config = ConfigDict(extra="allow")

    current_stage: WorkflowStage = "pending"
    percent_complete: float = 0.0
    message: str = "Waiting to start..."
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None


class WorkflowConfig(BaseModel):
    """
    User-supplied preferences that influence how the pipeline behaves.
    Passed in before the workflow begins.
    """
    model_config = ConfigDict(extra="allow")

    business_goal: Optional[str] = None
    preferred_chart_types: List[ChartType] = Field(default_factory=list)
    max_visualizations: int = Field(default=8, ge=1, le=20)
    quality_threshold: float = Field(
        default=40.0, ge=0.0, le=100.0,
        description="Minimum quality score to proceed past validation",
    )
    use_ai_analysis: bool = True
    output_format: Literal["twb", "twbx"] = "twb"


class WorkflowState(BaseModel):
    """
    The single mutable object that travels through every node in the
    LangGraph pipeline.  Each node reads what it needs, does its work,
    and writes its results back into this state.
    """
    model_config = ConfigDict(extra="allow")

    # --- identifiers -------------------------------------------------------
    run_id: str = Field(default_factory=lambda: str(uuid4()))

    # --- user inputs -------------------------------------------------------
    file_path: Optional[str] = None
    config: WorkflowConfig = Field(default_factory=WorkflowConfig)

    # --- data artifacts produced by each stage -----------------------------
    dataframe_json: Optional[str] = None  # serialised df (orient="split")
    dataset_schema: Optional[DatasetSchema] = None
    quality_report: Optional[QualityReport] = None
    profile_report: Optional[ProfileReport] = None
    analysis_result: Optional[AnalysisResult] = None
    dashboard_spec: Optional[DashboardSpec] = None
    workbook_spec: Optional[WorkbookSpec] = None
    output_path: Optional[str] = None

    # --- tracking ----------------------------------------------------------
    progress: WorkflowProgress = Field(default_factory=WorkflowProgress)
    errors: List[WorkflowError] = Field(default_factory=list)

    @property
    def has_fatal_error(self) -> bool:
        """True when any non-recoverable error has been recorded."""
        return any(not e.recoverable for e in self.errors)
