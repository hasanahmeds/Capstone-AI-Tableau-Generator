"""
5 NORMAL + 5 EDGE unit tests for DashboardAnalyzer
(with mocked API responses)

Run :
    pytest -s -q test_dashboard_analyzer.py
"""
# Necessary imports
from __future__ import annotations

import sys
import types
import shutil
from pathlib import Path

import pandas as pd
import pytest


# Safety stub: dashboard_analyzer imports prompt_templates.
# this stub prevents ImportError and still matches expected prompt format if the prompts are missing.

if "prompt_templates" not in sys.modules:
    stub = types.ModuleType("prompt_templates")

    def dataset_overview_prompt(profile_summary: str) -> dict:
        return {"system": "stub-system", "user": f"PROFILE:\n{profile_summary}"}

    def kpi_recommendation_prompt(profile_summary: str, dataset_domain: str, column_roles: dict) -> dict:
        return {
            "system": "stub-system",
            "user": f"DOMAIN={dataset_domain}\nROLES={column_roles}\nPROFILE:\n{profile_summary}",
        }

    stub.dataset_overview_prompt = dataset_overview_prompt
    stub.kpi_recommendation_prompt = kpi_recommendation_prompt
    sys.modules["prompt_templates"] = stub

from dashboard_analyzer import DashboardAnalyzer



# Fixture: copying the original train.csv into a temporary test folder and returns 
#           its path so unit tests can safely use the dataset without modifying the original file.

@pytest.fixture
def train_csv_path(tmp_path: Path) -> Path:
    # Looks for train.csv in your current project folder
    src = Path(__file__).parent / "train.csv"   # rename your file to train.csv
    if not src.exists():
        pytest.skip(f"train.csv not found at: {src}")

    dst = tmp_path / "train.csv"
    shutil.copy(src, dst)
    return dst



# Mock API responses (LLM outputs)

#These functions return mocked LLM responses for dataset overview and KPI recommendations, 
#   allowing unit tests to simulate API results without calling the real AI service

def _mock_overview() -> dict:
    return {
        "dataset_domain": "Retail Sales (MOCK)",
        "dataset_description": "Mocked dataset overview for testing.",
        "grain": "One row per order line (mocked).",
        "time_coverage": {"has_time_dimension": True, "time_column": "Order Date", "estimated_range": "2014-2017"},
        "key_entities": [{"entity_name": "Customer", "id_column": "Customer ID", "name_column": "Customer Name"}],
        "column_roles": {
            "dimensions": ["Category", "Sub-Category", "Segment", "Region"],
            "measures": ["Sales"],
            "temporal": ["Order Date", "Ship Date"],
            "identifiers": ["Order ID", "Customer ID", "Product ID"],
            "geographic": ["Country", "City", "State", "Postal Code"],
        },
        "notable_observations": ["Mocked observation 1", "Mocked observation 2"],
    }


def _mock_kpis() -> dict:
    return {
        "primary_kpis": [
            {"name": "Total Sales", "formula": "SUM([Sales])", "source_columns": ["Sales"], "format": "$#,##0"}
        ],
        "secondary_metrics": [
            {"name": "Average Sales", "formula": "AVG([Sales])", "source_columns": ["Sales"], "format": "$#,##0.00"}
        ],
        "trend_metrics": [
            {"name": "Monthly Sales Trend", "formula": "SUM([Sales])", "time_grain": "month",
             "source_columns": ["Order Date", "Sales"]}
        ],
        "comparative_metrics": [
            {"name": "Sales by Category", "measure": "SUM([Sales])", "compare_by": "Category",
             "source_columns": ["Category", "Sales"]}
        ],
    }



# 5 NORMAL TEST CASES

# 1 verifying that the DashboardAnalyzer successfully loads a CSV file into a pandas DataFrame and 
# correctly detects and parses date columns like Order Date and Ship Date. 
# It checks that these columns exist and confirms their data type is converted to datetime64. 
# This ensures the data loading and automatic date parsing functionality works correctly.

def test_normal_1_load_data_csv_and_parse_dates(train_csv_path: Path):
    print("\n================ NORMAL 1: load_data() CSV + date parsing ================")
    print("INPUT:", str(train_csv_path))
    print("EXPECTED: DataFrame loads; Order Date / Ship Date parsed to datetime64")

    analyzer = DashboardAnalyzer()  # no API key -> rule-based is fine
    df = analyzer.load_data(str(train_csv_path))

    print("ACTUAL: df.shape =", df.shape)
    print("ACTUAL: has 'Order Date' =", "Order Date" in df.columns)
    print("ACTUAL: has 'Ship Date'  =", "Ship Date" in df.columns)

    assert "Order Date" in df.columns
    assert "Ship Date" in df.columns

    print("ACTUAL: Order Date dtype =", df["Order Date"].dtype)
    print("ACTUAL: Ship Date  dtype =", df["Ship Date"].dtype)

    assert pd.api.types.is_datetime64_any_dtype(df["Order Date"])
    assert pd.api.types.is_datetime64_any_dtype(df["Ship Date"])

    print("RESULT: PASS ")
    

# 2 verifies that the analyze() function correctly uses the mocked LLM response instead of calling the real API. 
# It replaces the _call_llm method with a mock function using monkeypatch and checks that the returned overview matches the mocked data exactly. 
# This ensures the LLM integration logic works correctly without requiring a real API connection.
def test_normal_2_analyze_with_mocked_llm(train_csv_path: Path, monkeypatch: pytest.MonkeyPatch):
    print("\n================ NORMAL 2: analyze() with MOCKED LLM ================")
    print("INPUT:", str(train_csv_path))
    print("EXPECTED: analyze() returns mocked overview dict exactly")

    analyzer = DashboardAnalyzer(api_key="fake-key", endpoint="fake", model="fake-model")
    analyzer.load_data(str(train_csv_path))

    mocked = _mock_overview()
    monkeypatch.setattr(analyzer, "_call_llm", lambda prompt: mocked)

    result = analyzer.analyze()

    print("ACTUAL: dataset_domain =", result.get("dataset_domain"))
    print("ACTUAL: measures =", result.get("column_roles", {}).get("measures"))
    print("ACTUAL: returned == mocked =", result == mocked)

    assert result == mocked
    print("RESULT: PASS ")

# 3 verifies that the recommend_kpis() function correctly retrieves KPI recommendations using a mocked LLM response instead of calling the real API.
#  It uses monkeypatch to simulate two LLM calls—one for dataset overview and one for KPI generation—and 
# checks that the returned KPIs match the mocked data. This ensures the KPI recommendation logic works correctly and includes expected metrics 
# like Total Sales.

def test_normal_3_recommend_kpis_with_mocked_llm(train_csv_path: Path, monkeypatch: pytest.MonkeyPatch):
    print("\n================ NORMAL 3: recommend_kpis() with MOCKED LLM ================")
    print("INPUT:", str(train_csv_path))
    print("EXPECTED: recommend_kpis() returns mocked KPI dict; includes 'Total Sales'")

    analyzer = DashboardAnalyzer(api_key="fake-key", endpoint="fake", model="fake-model")
    analyzer.load_data(str(train_csv_path))

    mocked_overview = _mock_overview()
    mocked_kpis = _mock_kpis()

    calls = {"n": 0}

    def _mock_call_llm(_prompt: dict) -> dict:
        calls["n"] += 1
        return mocked_overview if calls["n"] == 1 else mocked_kpis

    monkeypatch.setattr(analyzer, "_call_llm", _mock_call_llm)

    kpis = analyzer.recommend_kpis()

    print("ACTUAL: primary_kpis =", kpis.get("primary_kpis"))
    assert kpis == mocked_kpis
    assert kpis["primary_kpis"][0]["name"] == "Total Sales"

    print("RESULT: PASS ")

# 4 This test verifies that the analyze() function works correctly without an API key by using the built-in rule-based fallback logic 
# instead of the LLM. It checks that the function still returns a valid overview containing column roles and dataset information. 
# This ensures the system can analyze datasets even when the AI service is unavailable.
def test_normal_4_analyze_rule_based_without_api_key(train_csv_path: Path):
    print("\n================ NORMAL 4: analyze() rule-based (no API key) ================")
    print("INPUT:", str(train_csv_path))
    print("EXPECTED: analyze() returns fallback overview with column_roles")

    analyzer = DashboardAnalyzer()  # no api_key -> fallback only
    analyzer.load_data(str(train_csv_path))
    overview = analyzer.analyze()

    print("ACTUAL: dataset_domain =", overview.get("dataset_domain"))
    print("ACTUAL: overview keys =", list(overview.keys()))
    print("ACTUAL: column_roles keys =", list(overview.get("column_roles", {}).keys()))

    assert "column_roles" in overview
    assert isinstance(overview["column_roles"], dict)

    print("RESULT: PASS ")

#verifies that the recommend_kpis() function generates KPI recommendations using the fallback rule-based logic when no API key is provided. 
# It confirms that the output contains a valid KPI dictionary and includes default metrics like Total Records. 
# This ensures KPI generation works reliably without depending on the AI API.
def test_normal_5_recommend_kpis_rule_based_without_api_key(train_csv_path: Path):
    print("\n================ NORMAL 5: recommend_kpis() rule-based (no API key) ================")
    print("INPUT:", str(train_csv_path))
    print("EXPECTED: KPI dict returned; contains 'Total Records' primary KPI")

    analyzer = DashboardAnalyzer()  # fallback only
    analyzer.load_data(str(train_csv_path))
    kpis = analyzer.recommend_kpis()

    primary_names = [k.get("name") for k in kpis.get("primary_kpis", [])]

    print("ACTUAL: KPI keys =", list(kpis.keys()))
    print("ACTUAL: primary KPI names (first 10) =", primary_names[:10])

    assert "primary_kpis" in kpis
    assert any(name == "Total Records" for name in primary_names)

    print("RESULT: PASS ")

#...........................................................................................................................................................................................................................

# 5 EDGE TEST CASES

# 1 This test checks that calling analyze() before loading any dataset is handled safely. 
# It verifies that the code raises a RuntimeError with an appropriate message instead of crashing silently. 
# This ensures the analyzer enforces the correct workflow: load data first, then analyze.
def test_edge_1_analyze_without_load_raises():
    print("\n================ EDGE 1: analyze() without load_data() ================")
    print("EXPECTED: RuntimeError ('No data loaded...')")

    analyzer = DashboardAnalyzer()
    with pytest.raises(RuntimeError) as e:
        analyzer.analyze()

    print("ACTUAL: Raised =", type(e.value).__name__)
    print("ACTUAL: Message =", str(e.value))
    print("RESULT: PASS ")

#ensures that recommend_kpis() cannot run without a dataset being loaded first. 
# It confirms that a RuntimeError is raised when the method is called without load_data(). 
# This prevents generating KPIs from an empty or undefined dataset.
def test_edge_2_recommend_kpis_without_load_raises():
    print("\n================ EDGE 2: recommend_kpis() without load_data() ================")
    print("EXPECTED: RuntimeError ('No data loaded...')")

    analyzer = DashboardAnalyzer()
    with pytest.raises(RuntimeError) as e:
        analyzer.recommend_kpis()

    print("ACTUAL: Raised =", type(e.value).__name__)
    print("ACTUAL: Message =", str(e.value))
    print("RESULT: PASS ")

# 3 validates error handling when the input file path is invalid or missing. 
# It checks that load_data() raises a FileNotFoundError when the file does not exist. 
# This ensures the system fails safely and gives a clear error instead of proceeding with invalid input.
def test_edge_3_load_data_missing_file_raises(tmp_path: Path):
    print("\n================ EDGE 3: load_data() missing file ================")
    print("EXPECTED: FileNotFoundError")

    analyzer = DashboardAnalyzer()
    missing = tmp_path / "not_here.csv"

    with pytest.raises(FileNotFoundError) as e:
        analyzer.load_data(str(missing))

    print("ACTUAL: Raised =", type(e.value).__name__)
    print("ACTUAL: Message =", str(e.value))
    print("RESULT: PASS ")

# 4 test verifies that load_data() rejects unsupported file formats like .txt. 
# It ensures the function raises a ValueError when an invalid extension is used. 
# This protects the pipeline from trying to parse unsupported files and producing incorrect results.
def test_edge_4_load_data_unsupported_extension_raises(tmp_path: Path):
    print("\n================ EDGE 4: load_data() unsupported extension ================")
    print("EXPECTED: ValueError (Unsupported format)")

    analyzer = DashboardAnalyzer()
    bad = tmp_path / "data.txt"
    bad.write_text("hello")

    with pytest.raises(ValueError) as e:
        analyzer.load_data(str(bad))

    print("ACTUAL: Raised =", type(e.value).__name__)
    print("ACTUAL: Message =", str(e.value))
    print("RESULT: PASS ")

# 5 simulates an LLM/API failure by forcing _call_llm() to raise an exception. 
# It verifies that analyze() still completes successfully using the rule-based fallback overview. 
# This ensures the analyzer remains reliable even if the AI service is down or unavailable.
def test_edge_5_analyze_falls_back_when_llm_throws(train_csv_path: Path, monkeypatch: pytest.MonkeyPatch):
    print("\n================ EDGE 5: analyze() fallback when LLM fails ================")
    print("INPUT:", str(train_csv_path))
    print("EXPECTED: LLM error happens, analyze() returns rule-based fallback overview")

    analyzer = DashboardAnalyzer(api_key="fake-key", endpoint="fake", model="fake-model")
    analyzer.load_data(str(train_csv_path))

    def _boom(_prompt: dict):
        raise RuntimeError("Simulated LLM outage")

    monkeypatch.setattr(analyzer, "_call_llm", _boom)

    overview = analyzer.analyze()

    print("ACTUAL: dataset_domain =", overview.get("dataset_domain"))
    print("ACTUAL: column_roles keys =", list(overview.get("column_roles", {}).keys()))

    # DashboardAnalyzer's fallback sets dataset_domain to something with "fallback"
    assert "fallback" in (overview.get("dataset_domain") or "").lower()
    assert "column_roles" in overview

    print("RESULT: PASS")

   

