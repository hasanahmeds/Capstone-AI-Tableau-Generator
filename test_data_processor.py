import pandas as pd
import pytest

from data_processor import DataProcessor, DataLoadError, DataValidationError


def _make_csv(tmp_path, df: pd.DataFrame, name: str = "data.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def _make_excel(tmp_path, df: pd.DataFrame, name: str = "data.xlsx"):
    p = tmp_path / name
    df.to_excel(p, index=False)
    return p

# 6 NORMAL CASES 


def test_normal_1_load_csv_success(tmp_path):
    print("\n=== NORMAL 1: load() CSV ===")
    df_in = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    csv_path = _make_csv(tmp_path, df_in, "ok.csv")

    dp = DataProcessor()
    df = dp.load(csv_path)

    print("Input file:", csv_path)
    print("Loaded df:\n", df)
    print("df.shape:", df.shape)
    print("df.columns:", list(df.columns))

    assert df.shape == (3, 2)
    assert list(df.columns) == ["a", "b"]


def test_normal_2_load_excel_success(tmp_path):
    print("\n=== NORMAL 2: load() Excel ===")
    df_in = pd.DataFrame({"x": [10, 20], "y": ["A", "B"]})
    xlsx_path = _make_excel(tmp_path, df_in, "ok.xlsx")

    dp = DataProcessor()
    df = dp.load(xlsx_path)

    print("Input file:", xlsx_path)
    print("Loaded df:\n", df)
    print("df.shape:", df.shape)
    print("df.columns:", list(df.columns))

    assert df.shape == (2, 2)
    assert list(df.columns) == ["x", "y"]


def test_normal_3_validate_infers_types_numeric_categorical_datetime_boolean_text():
    print("\n=== NORMAL 3: validate() type inference ===")

    # must have >50 unique long 
    n = 60
    df_in = pd.DataFrame(
        {
            "num": list(range(1, n + 1)),
            "cat": ["a"] * (n // 2) + ["b"] * (n - n // 2),
            "dt": pd.date_range("2024-01-01", periods=n, freq="D").astype(str).tolist(),
            "boolish": [0, 1] * (n // 2),
            "text": [f"row_{i}_" + ("x" * 80) for i in range(n)],
        }
    )

    dp = DataProcessor()
    schema = dp.validate(df_in, source_filename="sample.csv")

    print("schema.dataset_id:", schema.dataset_id)
    print("schema.row_count:", schema.row_count)
    print("schema.column_count:", schema.column_count)
    print("numeric_columns:", schema.numeric_columns)
    print("categorical_columns:", schema.categorical_columns)
    print("datetime_columns:", schema.datetime_columns)
    print("boolean_columns:", schema.boolean_columns)
    print("text_columns:", schema.text_columns)
    print("unknown_columns:", schema.unknown_columns)

    for col in schema.columns:
        print(
            f"  - {col.name}: semantic_type={col.semantic_type}, dtype_raw={col.dtype_raw}, "
            f"missing={col.missing_count} ({col.missing_ratio:.3f}), unique={col.unique_count}, "
            f"sample_values={col.sample_values}"
        )

    assert schema.row_count == n
    assert schema.column_count == 5
    assert "num" in schema.numeric_columns
    assert "cat" in schema.categorical_columns
    assert "dt" in schema.datetime_columns
    assert "boolish" in schema.boolean_columns
    assert "text" in schema.text_columns


def test_normal_4_assess_quality_detects_missing_duplicates_and_outliers():
    print("\n=== NORMAL 4: assess_quality() missing + duplicates + outliers ===")
    df = pd.DataFrame(
        {
            "value": [1, 2, 3, 1000],
            "city": ["A", "B", "B", None],
        }
    )
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    dp = DataProcessor()
    schema = dp.validate(df, source_filename="q.csv")
    report = dp.assess_quality(df, schema)

    print("missing.by_column:", report.missing.by_column)
    print("missing.total_missing:", report.missing.total_missing)
    print("missing.missing_ratio_overall:", report.missing.missing_ratio_overall)
    print("duplicates.duplicate_row_count:", report.duplicates.duplicate_row_count)
    print("duplicates.duplicate_row_ratio:", report.duplicates.duplicate_row_ratio)
    print("empty_columns:", report.empty_columns)
    print("constant_columns:", report.constant_columns)
    print("quality_score:", report.quality_score)
    print("warnings:", report.warnings)
    print("errors:", report.errors)

    if report.outliers is not None:
        print("outliers.method:", report.outliers.method)
        print("outliers.by_column:", report.outliers.by_column)
        print("outliers.bounds_by_column:", report.outliers.bounds_by_column)

    assert report.missing.total_missing >= 1
    assert "city" in report.missing.by_column
    assert report.duplicates.duplicate_row_count >= 1
    assert report.duplicates.duplicate_row_ratio > 0
    assert report.outliers is not None
    assert "value" in report.outliers.bounds_by_column


def test_normal_5_profile_builds_numeric_and_categorical_profiles():
    print("\n=== NORMAL 5: profile() numeric + categorical stats ===")
    df = pd.DataFrame({"num": [10, 20, 30, 40], "cat": ["x", "x", "y", "y"]})

    dp = DataProcessor()
    schema = dp.validate(df, source_filename="p.csv")
    prof = dp.profile(df, schema)

    print("profile.dataset_id:", prof.dataset_id)
    print("profile.generated_at:", prof.generated_at)
    for cp in prof.column_profiles:
        print(f"  - {cp.name} ({cp.semantic_type}) missing={cp.missing_count} ratio={cp.missing_ratio:.3f}")
        print("    numeric:", cp.numeric)
        print("    categorical:", cp.categorical)
        print("    datetime:", cp.datetime)
        print("    text:", cp.text)

    assert prof.dataset_id == schema.dataset_id
    assert len(prof.column_profiles) == 2


def test_normal_6_validate_preview_rows_and_sample_values_present():
    print("\n=== NORMAL 6: validate() preview_rows + sample_values ===")
    df = pd.DataFrame({"a": [1, None, 3], "b": ["  x ", "y", None]})

    dp = DataProcessor()
    schema = dp.validate(df, source_filename="prev.csv")

    print("preview_rows (len={}):".format(len(schema.preview_rows or [])))
    print(schema.preview_rows)

    for col in schema.columns:
        print(f"  - {col.name}: sample_values={col.sample_values}, missing={col.missing_count}")

    assert schema.preview_rows is not None
    assert len(schema.preview_rows) <= 5
    col_a = next(c for c in schema.columns if c.name == "a")
    assert isinstance(col_a.sample_values, list)


# 4 EDGE CASES 


def test_edge_1_load_missing_file_raises(tmp_path):
    print("\n=== EDGE 1: load() missing file ===")
    missing = tmp_path / "does_not_exist.csv"
    dp = DataProcessor()
    with pytest.raises(DataLoadError) as e:
        dp.load(missing)
    print("Raised:", type(e.value).__name__)
    print("Message:", str(e.value))


def test_edge_2_load_unsupported_extension_raises(tmp_path):
    print("\n=== EDGE 2: load() unsupported extension ===")
    p = tmp_path / "bad.txt"
    p.write_text("hello")
    dp = DataProcessor()
    with pytest.raises(DataLoadError) as e:
        dp.load(p)
    print("Raised:", type(e.value).__name__)
    print("Message:", str(e.value))


def test_edge_3_load_empty_dataframe_file_raises(tmp_path):
    print("\n=== EDGE 3: load() empty CSV (header only) ===")
    p = tmp_path / "empty.csv"
    p.write_text("a,b,c\n")

    dp = DataProcessor()
    with pytest.raises(DataLoadError) as e:
        dp.load(p)

    print("Raised:", type(e.value).__name__)
    print("Message:", str(e.value))


def test_edge_4_validate_all_nan_rows_raises():
    print("\n=== EDGE 4: validate() all-NaN rows ===")
    df = pd.DataFrame({"a": [None, None], "b": [None, None]})
    dp = DataProcessor()
    with pytest.raises(DataValidationError) as e:
        dp.validate(df, source_filename="nan.csv")

    print("Raised:", type(e.value).__name__)
    print("Message:", str(e.value))