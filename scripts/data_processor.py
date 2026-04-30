"""
DataProcessor: Data Processing Pipeline (Section 3.1)
=====================================================
Loads, validates, and assesses quality of CSV/Excel files.
Produces structured Pydantic outputs: DatasetSchema, QualityReport, ProfileReport.

Methods:
    - load(file_path)                    → pd.DataFrame
    - validate(df)                       → DatasetSchema
    - assess_quality(df, dataset_schema) → QualityReport
    - profile(df, dataset_schema)        → ProfileReport
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────
# Import Pydantic schemas from your schemas module
# ─────────────────────────────────────────────────────────────
from scripts.schemas import (
    ColumnSchema,
    DatasetSchema,
    TypeIssue,
    MissingValueReport,
    DuplicateReport,
    OutlierReport,
    QualityReport,
    NumericProfile,
    CategoricalProfile,
    DatetimeProfile,
    TextProfile,
    ColumnProfile,
    ProfileReport,
    ColumnSemanticType,
    FileFormat,
    Severity,
)

# ─────────────────────────────────────────────────────────────
# CONFIGURATION CONSTANTS
# ─────────────────────────────────────────────────────────────

MIN_ROWS = 1
MAX_COLUMNS = 500
CATEGORICAL_CARDINALITY_THRESHOLD = 50
TEXT_AVG_LENGTH_THRESHOLD = 50
MISSING_RATIO_WARNING = 0.05
MISSING_RATIO_HIGH = 0.30
DUPLICATE_RATIO_WARNING = 0.05
IQR_MULTIPLIER = 1.5
TOP_K_CATEGORIES = 10
PREVIEW_ROWS = 5
SAMPLE_VALUES_COUNT = 5

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

# ─────────────────────────────────────────────────────────────
# LOGGER
# ─────────────────────────────────────────────────────────────

from scripts.logger_config import get_logger

logger = get_logger("data_processor")

# ─────────────────────────────────────────────────────────────
# CUSTOM EXCEPTIONS
# ─────────────────────────────────────────────────────────────


class DataLoadError(Exception):
    """Raised when a file cannot be loaded."""


class DataValidationError(Exception):
    """Raised when the loaded data fails minimum requirements."""


# ─────────────────────────────────────────────────────────────
# DataProcessor CLASS
# ─────────────────────────────────────────────────────────────


class DataProcessor:
    """
    End-to-end data processing pipeline for the AI Tableau Generator.

    Usage:
        processor = DataProcessor()
        df = processor.load("sales_data.csv")
        schema = processor.validate(df, source_filename="sales_data.csv")
        quality = processor.assess_quality(df, schema)
        profile = processor.profile(df, schema)
    """

    # ── LOAD ────────────────────────────────────────────────

    def load(
        self,
        file_path: Union[str, Path],
        *,
        sheet_name: Union[str, int, None] = 0,
        encoding: str = "utf-8",
    ) -> pd.DataFrame:
        """
        Load a CSV or Excel file into a pandas DataFrame.

        Args:
            file_path: Path to the .csv / .xlsx / .xls file.
            sheet_name: Sheet to load for Excel files (default: first sheet).
            encoding: Text encoding for CSV files (default utf-8; falls back to latin-1).

        Returns:
            pd.DataFrame with the raw data.

        Raises:
            DataLoadError on unsupported format, missing file, or parse failure.
        """
        file_path = Path(file_path)
        logger.info("Loading file: %s", file_path)

        # --- existence check ---
        if not file_path.exists():
            raise DataLoadError(f"File not found: {file_path}")

        # --- extension check ---
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise DataLoadError(
                f"Unsupported file format '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # --- read ---
        try:
            if ext == ".csv":
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                except UnicodeDecodeError:
                    logger.warning(
                        "UTF-8 decode failed; retrying with latin-1 encoding."
                    )
                    df = pd.read_csv(file_path, encoding="latin-1")
            else:  # .xlsx / .xls
                df = pd.read_excel(file_path, sheet_name=sheet_name)
        except Exception as exc:
            raise DataLoadError(f"Failed to parse '{file_path.name}': {exc}") from exc

        # --- basic sanity ---
        if df.empty:
            raise DataLoadError(
                "File loaded but DataFrame is empty (0 rows × 0 cols)."
            )

        # Strip leading/trailing whitespace from column names
        df.columns = [str(c).strip() for c in df.columns]

        logger.info(
            "Loaded %d rows × %d columns from '%s'",
            len(df),
            len(df.columns),
            file_path.name,
        )
        return df

    # ── VALIDATE ────────────────────────────────────────────

    def validate(
        self,
        df: pd.DataFrame,
        *,
        source_filename: Optional[str] = None,
        dataset_name: Optional[str] = None,
    ) -> DatasetSchema:
        """
        Detect column types and build a DatasetSchema for the DataFrame.

        Validation checks:
            - Minimum row count (>= MIN_ROWS)
            - Maximum column count (<= MAX_COLUMNS)
            - No fully-empty DataFrame

        Args:
            df: The loaded DataFrame.
            source_filename: Original filename (for metadata).
            dataset_name: Friendly name for the dataset.

        Returns:
            DatasetSchema populated with column metadata.

        Raises:
            DataValidationError if minimum requirements are not met.
        """
        logger.info(
            "Validating DataFrame (%d rows × %d cols)", len(df), len(df.columns)
        )

        # ── minimum-requirement checks ──
        if df.shape[0] < MIN_ROWS:
            raise DataValidationError(
                f"Dataset has {df.shape[0]} rows; minimum required is {MIN_ROWS}."
            )
        if df.shape[1] > MAX_COLUMNS:
            raise DataValidationError(
                f"Dataset has {df.shape[1]} columns; maximum allowed is {MAX_COLUMNS}."
            )
        if df.dropna(how="all").empty:
            raise DataValidationError("All rows are empty (NaN). Cannot proceed.")

        # ── determine file format ──
        file_format: FileFormat = "csv"
        if source_filename:
            ext = Path(source_filename).suffix.lower()
            if ext in (".xlsx", ".xls"):
                file_format = "excel"

        # ── build column schemas ──
        columns: List[ColumnSchema] = []
        type_buckets: Dict[ColumnSemanticType, List[str]] = {
            "numeric": [],
            "categorical": [],
            "datetime": [],
            "boolean": [],
            "text": [],
            "unknown": [],
        }

        for col_name in df.columns:
            col_series = df[col_name]
            sem_type = self._infer_semantic_type(col_series)

            missing_cnt = int(col_series.isna().sum())
            total = len(col_series)
            unique_cnt = int(col_series.nunique(dropna=True))

            # Sample non-null values
            non_null = col_series.dropna()
            sample_vals = (
                non_null.head(SAMPLE_VALUES_COUNT).tolist()
                if len(non_null) > 0
                else []
            )
            sample_vals = [self._to_python_native(v) for v in sample_vals]

            col_schema = ColumnSchema(
                name=col_name,
                dtype_raw=str(col_series.dtype),
                semantic_type=sem_type,
                missing_count=missing_cnt,
                missing_ratio=round(missing_cnt / total, 4) if total > 0 else 0.0,
                unique_count=unique_cnt,
                is_constant=(unique_cnt <= 1),
                is_empty=(non_null.shape[0] == 0),
                sample_values=sample_vals,
            )
            columns.append(col_schema)
            type_buckets[sem_type].append(col_name)

        # ── preview rows ──
        preview = (
            df.head(PREVIEW_ROWS).replace({np.nan: None}).to_dict(orient="records")
        )

        schema = DatasetSchema(
            name=dataset_name or "uploaded_dataset",
            file_format=file_format,
            source_filename=source_filename,
            row_count=len(df),
            column_count=len(df.columns),
            columns=columns,
            numeric_columns=type_buckets["numeric"],
            categorical_columns=type_buckets["categorical"],
            datetime_columns=type_buckets["datetime"],
            boolean_columns=type_buckets["boolean"],
            text_columns=type_buckets["text"],
            unknown_columns=type_buckets["unknown"],
            preview_rows=preview,
        )

        logger.info(
            "Validation complete — numeric: %d, categorical: %d, datetime: %d, "
            "boolean: %d, text: %d, unknown: %d",
            len(schema.numeric_columns),
            len(schema.categorical_columns),
            len(schema.datetime_columns),
            len(schema.boolean_columns),
            len(schema.text_columns),
            len(schema.unknown_columns),
        )
        return schema

    # ── ASSESS QUALITY ──────────────────────────────────────

    def assess_quality(
        self,
        df: pd.DataFrame,
        schema: DatasetSchema,
    ) -> QualityReport:
        """
        Run quality checks on the DataFrame and return a QualityReport.

        Checks performed:
            1. Missing values (per-column and overall)
            2. Duplicate rows
            3. Outliers via IQR for numeric columns
            4. Empty columns (100% NaN)
            5. Constant columns (single unique value)
            6. Type issues (e.g., numeric column with non-numeric strings)
            7. Composite quality score (0-100)

        Args:
            df: The loaded DataFrame.
            schema: DatasetSchema produced by validate().

        Returns:
            QualityReport with all findings.
        """
        logger.info("Assessing data quality for dataset '%s'", schema.dataset_id)

        warnings: List[str] = []
        errors: List[str] = []

        # 1. MISSING VALUES
        missing_by_col: Dict[str, int] = {}
        for col_schema in schema.columns:
            if col_schema.missing_count > 0:
                missing_by_col[col_schema.name] = col_schema.missing_count

        total_cells = schema.row_count * schema.column_count
        total_missing = int(df.isna().sum().sum())
        missing_ratio_overall = (
            round(total_missing / total_cells, 4) if total_cells > 0 else 0.0
        )

        missing_report = MissingValueReport(
            by_column=missing_by_col,
            total_missing=total_missing,
            missing_ratio_overall=missing_ratio_overall,
        )

        if missing_ratio_overall > MISSING_RATIO_WARNING:
            warnings.append(
                f"Overall missing ratio is {missing_ratio_overall:.1%} "
                f"(threshold: {MISSING_RATIO_WARNING:.0%})."
            )

        for col_schema in schema.columns:
            if col_schema.missing_ratio > MISSING_RATIO_HIGH:
                warnings.append(
                    f"Column '{col_schema.name}' has "
                    f"{col_schema.missing_ratio:.1%} missing values."
                )

        # 2. DUPLICATES
        dup_count = int(df.duplicated().sum())
        dup_ratio = (
            round(dup_count / schema.row_count, 4) if schema.row_count > 0 else 0.0
        )

        dup_report = DuplicateReport(
            duplicate_row_count=dup_count,
            duplicate_row_ratio=dup_ratio,
        )

        if dup_ratio > DUPLICATE_RATIO_WARNING:
            warnings.append(
                f"{dup_count} duplicate rows detected ({dup_ratio:.1%} of data)."
            )

        # 3. OUTLIERS (IQR)
        outlier_by_col: Dict[str, int] = {}
        bounds_by_col: Dict[str, Dict[str, float]] = {}

        for col_name in schema.numeric_columns:
            series = pd.to_numeric(df[col_name], errors="coerce").dropna()
            if series.empty:
                continue
            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - IQR_MULTIPLIER * iqr
            upper = q3 + IQR_MULTIPLIER * iqr
            outlier_mask = (series < lower) | (series > upper)
            n_outliers = int(outlier_mask.sum())
            if n_outliers > 0:
                outlier_by_col[col_name] = n_outliers
            bounds_by_col[col_name] = {
                "low": round(lower, 4),
                "high": round(upper, 4),
            }

        outlier_report = OutlierReport(
            method="iqr",
            by_column=outlier_by_col,
            bounds_by_column=bounds_by_col,
        )

        if outlier_by_col:
            total_outliers = sum(outlier_by_col.values())
            warnings.append(
                f"{total_outliers} outlier values detected across "
                f"{len(outlier_by_col)} numeric column(s)."
            )

        # 4. EMPTY COLUMNS
        empty_cols = [cs.name for cs in schema.columns if cs.is_empty]
        if empty_cols:
            errors.append(f"Fully empty columns: {empty_cols}")

        # 5. CONSTANT COLUMNS
        constant_cols = [
            cs.name for cs in schema.columns if cs.is_constant and not cs.is_empty
        ]
        if constant_cols:
            warnings.append(f"Constant (single-value) columns: {constant_cols}")

        # 6. TYPE ISSUES
        type_issues: List[TypeIssue] = []

        for col_name in schema.numeric_columns:
            non_numeric = pd.to_numeric(df[col_name], errors="coerce")
            coercion_failures = df[col_name].notna() & non_numeric.isna()
            n_bad = int(coercion_failures.sum())
            if n_bad > 0:
                bad_examples = (
                    df.loc[coercion_failures, col_name].head(5).tolist()
                )
                severity: Severity = (
                    "high" if n_bad / schema.row_count > 0.1 else "medium"
                )
                type_issues.append(
                    TypeIssue(
                        column=col_name,
                        issue="Non-numeric values in numeric column",
                        count=n_bad,
                        examples=bad_examples,
                        severity=severity,
                    )
                )

        for col_name in schema.datetime_columns:
            parsed = pd.to_datetime(
                df[col_name], errors="coerce", infer_datetime_format=True
            )
            coercion_failures = df[col_name].notna() & parsed.isna()
            n_bad = int(coercion_failures.sum())
            if n_bad > 0:
                bad_examples = (
                    df.loc[coercion_failures, col_name].head(5).tolist()
                )
                type_issues.append(
                    TypeIssue(
                        column=col_name,
                        issue="Unparseable datetime values",
                        count=n_bad,
                        examples=bad_examples,
                        severity="medium",
                    )
                )

        # 7. QUALITY SCORE (0-100)
        quality_score = self._compute_quality_score(
            schema=schema,
            missing_ratio=missing_ratio_overall,
            dup_ratio=dup_ratio,
            n_empty_cols=len(empty_cols),
            n_constant_cols=len(constant_cols),
            n_type_issues=len(type_issues),
        )

        report = QualityReport(
            dataset_id=schema.dataset_id,
            missing=missing_report,
            duplicates=dup_report,
            outliers=outlier_report,
            empty_columns=empty_cols,
            constant_columns=constant_cols,
            type_issues=type_issues,
            quality_score=quality_score,
            warnings=warnings,
            errors=errors,
        )

        logger.info("Quality score: %.1f / 100", quality_score)
        return report

    # ── PROFILE ─────────────────────────────────────────────

    def profile(
        self,
        df: pd.DataFrame,
        schema: DatasetSchema,
    ) -> ProfileReport:
        """
        Generate a statistical profile for every column.

        Produces per-column profiles:
            - NumericProfile   (count, mean, std, min, p25, median, p75, max)
            - CategoricalProfile (count, unique, top, freq, top_k)
            - DatetimeProfile  (count, min date, max date)
            - TextProfile      (count, avg/min/max string length)

        Args:
            df: The loaded DataFrame.
            schema: DatasetSchema produced by validate().

        Returns:
            ProfileReport with a ColumnProfile for each column.
        """
        logger.info("Profiling dataset '%s'", schema.dataset_id)

        col_profiles: List[ColumnProfile] = []
        notes: List[str] = []

        for col_schema in schema.columns:
            col_name = col_schema.name
            sem_type = col_schema.semantic_type
            series = df[col_name]

            cp = ColumnProfile(
                name=col_name,
                semantic_type=sem_type,
                missing_count=col_schema.missing_count,
                missing_ratio=col_schema.missing_ratio,
            )

            # ── Numeric ──
            if sem_type == "numeric":
                num_series = pd.to_numeric(series, errors="coerce").dropna()
                if not num_series.empty:
                    desc = num_series.describe()
                    cp.numeric = NumericProfile(
                        count=int(desc["count"]),
                        mean=self._safe_float(desc.get("mean")),
                        std=self._safe_float(desc.get("std")),
                        min=self._safe_float(desc.get("min")),
                        p25=self._safe_float(desc.get("25%")),
                        median=self._safe_float(desc.get("50%")),
                        p75=self._safe_float(desc.get("75%")),
                        max=self._safe_float(desc.get("max")),
                    )
                else:
                    cp.numeric = NumericProfile(count=0)
                    notes.append(
                        f"Column '{col_name}' is numeric but has no valid values."
                    )

            # ── Categorical ──
            elif sem_type == "categorical":
                non_null = series.dropna()
                vc = non_null.value_counts()
                top_k = [
                    {"value": self._to_python_native(idx), "count": int(cnt)}
                    for idx, cnt in vc.head(TOP_K_CATEGORIES).items()
                ]
                cp.categorical = CategoricalProfile(
                    count=int(non_null.shape[0]),
                    unique=int(vc.shape[0]),
                    top=(
                        self._to_python_native(vc.index[0]) if len(vc) > 0 else None
                    ),
                    freq=int(vc.iloc[0]) if len(vc) > 0 else None,
                    top_k=top_k,
                )

            # ── Boolean (treat like categorical with 2 values) ──
            elif sem_type == "boolean":
                non_null = series.dropna()
                vc = non_null.value_counts()
                top_k = [
                    {"value": self._to_python_native(idx), "count": int(cnt)}
                    for idx, cnt in vc.items()
                ]
                cp.categorical = CategoricalProfile(
                    count=int(non_null.shape[0]),
                    unique=int(vc.shape[0]),
                    top=(
                        self._to_python_native(vc.index[0]) if len(vc) > 0 else None
                    ),
                    freq=int(vc.iloc[0]) if len(vc) > 0 else None,
                    top_k=top_k,
                )

            # ── Datetime ──
            elif sem_type == "datetime":
                dt_series = pd.to_datetime(series, errors="coerce").dropna()
                if not dt_series.empty:
                    cp.datetime = DatetimeProfile(
                        count=int(dt_series.shape[0]),
                        min=str(dt_series.min().isoformat()),
                        max=str(dt_series.max().isoformat()),
                    )
                else:
                    cp.datetime = DatetimeProfile(count=0)
                    notes.append(
                        f"Column '{col_name}' is datetime but has no parseable values."
                    )

            # ── Text ──
            elif sem_type == "text":
                str_series = series.dropna().astype(str)
                if not str_series.empty:
                    lengths = str_series.str.len()
                    cp.text = TextProfile(
                        count=int(str_series.shape[0]),
                        avg_length=round(float(lengths.mean()), 2),
                        min_length=int(lengths.min()),
                        max_length=int(lengths.max()),
                    )
                else:
                    cp.text = TextProfile(count=0)

            # ── Unknown ──
            else:
                non_null = series.dropna()
                if not non_null.empty:
                    notes.append(
                        f"Column '{col_name}' has unknown type; "
                        f"{non_null.shape[0]} non-null values."
                    )

            col_profiles.append(cp)

        report = ProfileReport(
            dataset_id=schema.dataset_id,
            column_profiles=col_profiles,
            notes=notes,
        )

        logger.info(
            "Profiling complete — %d column profiles generated.", len(col_profiles)
        )
        return report

    # ─────────────────────────────────────────────────────────
    # PRIVATE / HELPER METHODS
    # ─────────────────────────────────────────────────────────

    def _infer_semantic_type(self, series: pd.Series) -> ColumnSemanticType:
        """
        Heuristically infer the semantic type of a pandas Series.

        Rules (applied in order):
            1. If all non-null values are bool         → "boolean"
            2. If dtype is datetime64 or parseable     → "datetime"
            3. If dtype is numeric (int/float)         → "numeric"
            4. If object/string dtype:
                a. >= 80% numeric coercion success     → "numeric"
                b. >= 80% datetime coercion success    → "datetime"
                c. unique <= CATEGORICAL_CARDINALITY   → "categorical"
                d. avg str length > TEXT_AVG_LENGTH     → "text"
                e. else                                → "categorical"
            5. Fallback                                → "unknown"
        """
        non_null = series.dropna()
        if non_null.empty:
            return "unknown"

        # 1. Boolean
        if pd.api.types.is_bool_dtype(series):
            return "boolean"
        if series.dtype == object:
            unique_vals = set(non_null.unique())
            if unique_vals <= {True, False, "True", "False", "true", "false", 0, 1}:
                if len(unique_vals) <= 2:
                    return "boolean"

        # 2. Datetime (native)
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # 3. Numeric (native)
        if pd.api.types.is_numeric_dtype(series):
            unique_vals = set(non_null.unique())
            if unique_vals <= {0, 1, 0.0, 1.0} and len(unique_vals) <= 2:
                return "boolean"
            return "numeric"

        # 4. Object / string heuristics
        if series.dtype == object or pd.api.types.is_string_dtype(series):
            n = len(non_null)

            # 4a. Numeric coercion
            numeric_coerced = pd.to_numeric(non_null, errors="coerce")
            if numeric_coerced.notna().sum() / n >= 0.80:
                return "numeric"

            # 4b. Datetime coercion
            try:
                dt_coerced = pd.to_datetime(
                    non_null, errors="coerce"
                )
                if dt_coerced.notna().sum() / n >= 0.80:
                    return "datetime"
            except Exception:
                pass

            # 4c / 4d. Categorical vs Text
            n_unique = non_null.nunique()
            avg_len = non_null.astype(str).str.len().mean()

            if n_unique <= CATEGORICAL_CARDINALITY_THRESHOLD:
                return "categorical"
            if avg_len > TEXT_AVG_LENGTH_THRESHOLD:
                return "text"

            return "categorical"

        # 5. Fallback
        return "unknown"

    @staticmethod
    def _compute_quality_score(
        schema: DatasetSchema,
        missing_ratio: float,
        dup_ratio: float,
        n_empty_cols: int,
        n_constant_cols: int,
        n_type_issues: int,
    ) -> float:
        """
        Compute a composite quality score from 0 to 100.

        Deductions:
            - Missing values:   up to -30 pts
            - Duplicates:       up to -20 pts
            - Empty columns:    -5 each  (capped at -15)
            - Constant columns: -3 each  (capped at -9)
            - Type issues:      -5 each  (capped at -15)
        """
        score = 100.0
        score -= min(missing_ratio * 100, 30.0)
        score -= min(dup_ratio * 100, 20.0)
        score -= min(n_empty_cols * 5, 15)
        score -= min(n_constant_cols * 3, 9)
        score -= min(n_type_issues * 5, 15)
        return round(max(score, 0.0), 2)

    @staticmethod
    def _safe_float(val: Any) -> Optional[float]:
        """Convert a value to float, returning None for non-finite."""
        if val is None:
            return None
        try:
            f = float(val)
            return f if np.isfinite(f) else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _to_python_native(val: Any) -> Any:
        """Convert numpy scalar types to Python-native for JSON serialisation."""
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, (np.bool_,)):
            return bool(val)
        if isinstance(val, (np.ndarray,)):
            return val.tolist()
        if isinstance(val, pd.Timestamp):
            return val.isoformat()
        return val


# ─────────────────────────────────────────────────────────────
# CONVENIENCE: run full pipeline
# ─────────────────────────────────────────────────────────────


def run_pipeline(
    file_path: Union[str, Path],
    *,
    dataset_name: Optional[str] = None,
) -> Tuple[pd.DataFrame, DatasetSchema, QualityReport, ProfileReport]:
    """
    Run the full DataProcessor pipeline on a single file.

    Returns:
        (df, schema, quality_report, profile_report)
    """
    processor = DataProcessor()
    df = processor.load(file_path)
    schema = processor.validate(
        df,
        source_filename=str(Path(file_path).name),
        dataset_name=dataset_name,
    )
    quality = processor.assess_quality(df, schema)
    profile = processor.profile(df, schema)
    return df, schema, quality, profile

'''
# ─────────────────────────────────────────────────────────────
# MAIN (demo / CLI smoke test)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import os

    df, schema, quality, profile = run_pipeline("train.csv", dataset_name="Train Dataset")

    os.makedirs("logs", exist_ok=True)
    with open("logs/output_results.json", "w") as f:
        f.write("===== DATASET SCHEMA =====\n")
        f.write(schema.model_dump_json(indent=2))
        f.write("\n\n===== QUALITY REPORT =====\n")
        f.write(quality.model_dump_json(indent=2))
        f.write("\n\n===== PROFILE REPORT =====\n")
        f.write(profile.model_dump_json(indent=2))

    print("Results saved to logs/output_results.json")
'''