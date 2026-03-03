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
ChartType = Literal["bar", "line", "scatter", "histogram", "box", "heatmap", "pie", "table"]


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
    parsed = pd.to_datetime(non_null, errors="coerce", infer_datetime_format=True)
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
        parsed = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
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
                s, errors="coerce", infer_datetime_format=True
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
    """Build a basic DashboardSpec with auto-recommended visualizations."""
    visuals: List[VisualizationSpec] = []

    # 1) Datetime + numeric → line chart
    if ds.datetime_columns and ds.numeric_columns:
        visuals.append(
            VisualizationSpec(
                chart_type="line",
                title=f"{ds.numeric_columns[0]} over time",
                x=ds.datetime_columns[0],
                y=ds.numeric_columns[0],
                aggregation="sum",
                rationale="Time trend if a datetime column exists.",
                confidence=0.8,
            )
        )

    # 2) At least 1 numeric → histogram
    if ds.numeric_columns:
        visuals.append(
            VisualizationSpec(
                chart_type="histogram",
                title=f"Distribution of {ds.numeric_columns[0]}",
                x=ds.numeric_columns[0],
                rationale="Shows distribution for the first numeric column.",
                confidence=0.75,
            )
        )

    # 3) At least 2 numeric → scatter
    if len(ds.numeric_columns) >= 2:
        visuals.append(
            VisualizationSpec(
                chart_type="scatter",
                title=f"{ds.numeric_columns[0]} vs {ds.numeric_columns[1]}",
                x=ds.numeric_columns[0],
                y=ds.numeric_columns[1],
                rationale="Shows relationship between two numeric columns.",
                confidence=0.7,
            )
        )

    # 4) Categorical + numeric → bar chart
    if ds.categorical_columns and ds.numeric_columns:
        visuals.append(
            VisualizationSpec(
                chart_type="bar",
                title=f"{ds.numeric_columns[0]} by {ds.categorical_columns[0]}",
                x=ds.categorical_columns[0],
                y=ds.numeric_columns[0],
                aggregation="sum",
                rationale="Aggregated comparison across categories.",
                confidence=0.72,
            )
        )

    return DashboardSpec(
        name="Auto Generated Dashboard",
        visuals=visuals,
        layout={"type": "grid", "columns": 2, "order": list(range(len(visuals)))},
    )