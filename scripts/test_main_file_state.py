#to run this file:
#python -m pytest -s test_main_file_state.py
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import app


# ============================================================
# Helper classes / functions
# ============================================================

class FakeUploadedFile:
    """Simple fake uploaded file object for testing."""
    def __init__(self, name: str, content: bytes, size: int = None):
        self.name = name
        self._content = content
        self.size = len(content) if size is None else size

    def getbuffer(self):
        return self._content


class FakeSessionState(dict):
    """Dict-like fake Streamlit session state with attribute access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
        else:
            raise AttributeError(key)


def print_test_banner(name: str):
    print("\n" + "=" * 70)
    print(f"TEST: {name}")
    print("=" * 70)


def print_result(status: str, details: str):
    print(details)
    print(f"\nRESULT: {status}")


@pytest.fixture
def fake_state(monkeypatch):
    """Replace Streamlit session_state with a controllable fake object."""
    state = FakeSessionState()
    monkeypatch.setattr(app.st, "session_state", state)
    return state


# ============================================================
# 4 NORMAL TEST CASES
# ============================================================

def test_normal_1_validate_valid_csv():
    print_test_banner("Normal Test 1 - Validate Proper CSV File")

    uploaded = FakeUploadedFile(
        name="sales.csv",
        content=b"col1,col2\n1,2\n3,4\n"
    )

    ok, err = app._validate_uploaded_file(uploaded)

    print_result(
        "PASS" if (ok is True and err == "") else "FAIL",
        f"Input File Name : {uploaded.name}\n"
        f"Input File Size : {uploaded.size} bytes\n"
        f"Validation OK   : {ok}\n"
        f"Error Message   : {err!r}"
    )

    assert ok is True
    assert err == ""


def test_normal_2_load_preview_csv():
    print_test_banner("Normal Test 2 - Load CSV Preview")

    df_original = pd.DataFrame({
        "Product": ["A", "B", "C"],
        "Sales": [100, 200, 300]
    })

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as tmp:
        df_original.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    try:
        df_loaded = app._load_preview(tmp_path)

        print_result(
            "PASS" if df_loaded.shape == (3, 2) else "FAIL",
            f"Temporary File  : {tmp_path}\n"
            f"Loaded Shape    : {df_loaded.shape}\n"
            f"Loaded Columns  : {list(df_loaded.columns)}\n"
            f"First Rows:\n{df_loaded.head().to_string(index=False)}"
        )

        assert isinstance(df_loaded, pd.DataFrame)
        assert df_loaded.shape == (3, 2)
        assert list(df_loaded.columns) == ["Product", "Sales"]
    finally:
        os.remove(tmp_path)


def test_normal_3_quick_quality_clean_data():
    print_test_banner("Normal Test 3 - Quick Quality on Clean Dataset")

    df = pd.DataFrame({
        "A": [1, 2, 3, 4],
        "B": ["x", "y", "z", "w"]
    })

    result = app._quick_quality(df)

    print_result(
        "PASS" if (result["rows"] == 4 and result["cols"] == 2 and result["missing"] == 0 and result["dupes"] == 0) else "FAIL",
        f"Rows            : {result['rows']}\n"
        f"Columns         : {result['cols']}\n"
        f"Missing Cells   : {result['missing']}\n"
        f"Missing %       : {result['missing_pct']}\n"
        f"Duplicate Rows  : {result['dupes']}\n"
        f"Duplicate %     : {result['dupe_pct']}\n"
        f"Quality Score   : {result['score']}"
    )

    assert result["rows"] == 4
    assert result["cols"] == 2
    assert result["missing"] == 0
    assert result["dupes"] == 0
    assert 0 <= result["score"] <= 100


def test_normal_4_init_state_sets_defaults(fake_state):
    print_test_banner("Normal Test 4 - Initialize Default State")

    app._init_state()

    print_result(
        "PASS" if (
            fake_state.page == "upload"
            and fake_state.uploaded_file_path is None
            and fake_state.uploaded_file_name is None
            and isinstance(fake_state.config, dict)
        ) else "FAIL",
        f"Page                : {fake_state.page}\n"
        f"Uploaded File Path  : {fake_state.uploaded_file_path}\n"
        f"Uploaded File Name  : {fake_state.uploaded_file_name}\n"
        f"DF Preview          : {fake_state.df_preview}\n"
        f"Quality Quick       : {fake_state.quality_quick}\n"
        f"Pipeline Result     : {fake_state.pipeline_result}\n"
        f"Pipeline Running    : {fake_state.pipeline_running}\n"
        f"Config              : {fake_state.config}"
    )

    assert fake_state.page == "upload"
    assert fake_state.uploaded_file_path is None
    assert fake_state.uploaded_file_name is None
    assert fake_state.df_preview is None
    assert fake_state.quality_quick is None
    assert fake_state.pipeline_result is None
    assert fake_state.pipeline_running is False
    assert fake_state.config["provider"] == "gemini"


# ============================================================
# 4 EDGE TEST CASES
# ============================================================

def test_edge_1_invalid_file_extension():
    print_test_banner("Edge Test 1 - Reject Invalid File Extension")

    uploaded = FakeUploadedFile(
        name="notes.txt",
        content=b"this is not a csv or excel file"
    )

    ok, err = app._validate_uploaded_file(uploaded)

    print_result(
        "PASS" if (ok is False and "Unsupported file type" in err) else "FAIL",
        f"Input File Name : {uploaded.name}\n"
        f"Input File Size : {uploaded.size} bytes\n"
        f"Validation OK   : {ok}\n"
        f"Error Message   : {err!r}"
    )

    assert ok is False
    assert "Unsupported file type" in err


def test_edge_2_file_too_large():
    print_test_banner("Edge Test 2 - Reject Oversized File")

    oversized_bytes = (app.MAX_FILE_SIZE_MB * 1024 * 1024) + 1
    uploaded = FakeUploadedFile(
        name="huge.csv",
        content=b"x",
        size=oversized_bytes
    )

    ok, err = app._validate_uploaded_file(uploaded)

    print_result(
        "PASS" if (ok is False and "max allowed" in err) else "FAIL",
        f"Input File Name : {uploaded.name}\n"
        f"Input File Size : {uploaded.size} bytes\n"
        f"Max Allowed MB  : {app.MAX_FILE_SIZE_MB}\n"
        f"Validation OK   : {ok}\n"
        f"Error Message   : {err!r}"
    )

    assert ok is False
    assert "max allowed" in err


def test_edge_3_quick_quality_with_missing_and_duplicates():
    print_test_banner("Edge Test 3 - Quality Check with Missing Values and Duplicates")

    df = pd.DataFrame({
        "A": [1, 1, None, 3],
        "B": ["x", "x", "y", "z"]
    })

    result = app._quick_quality(df)

    print_result(
        "PASS" if (result["missing"] == 1 and result["dupes"] == 1) else "FAIL",
        f"Input Data:\n{df.to_string(index=False)}\n\n"
        f"Rows            : {result['rows']}\n"
        f"Columns         : {result['cols']}\n"
        f"Missing Cells   : {result['missing']}\n"
        f"Missing %       : {result['missing_pct']}\n"
        f"Duplicate Rows  : {result['dupes']}\n"
        f"Duplicate %     : {result['dupe_pct']}\n"
        f"Quality Score   : {result['score']}"
    )

    assert result["missing"] == 1
    assert result["dupes"] == 1
    assert 0 <= result["score"] <= 100


def test_edge_4_state_navigation_and_reset(fake_state):
    print_test_banner("Edge Test 4 - Page Navigation and Reset State")

    app._init_state()
    app._go("configure")
    page_after_go = fake_state.page

    fake_state.extra_value = "temporary_data"
    count_before_reset = len(fake_state.keys())

    app._reset()
    count_after_reset = len(fake_state.keys())

    print_result(
        "PASS" if (page_after_go == "configure" and count_after_reset == 0) else "FAIL",
        f"Page After _go('configure') : {page_after_go}\n"
        f"Keys Before Reset          : {count_before_reset}\n"
        f"Keys After Reset           : {count_after_reset}\n"
        f"Remaining State            : {dict(fake_state)}"
    )

    assert page_after_go == "configure"
    assert count_after_reset == 0