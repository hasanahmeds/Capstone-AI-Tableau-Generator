"""
Streamlit UI for the AI-Powered Tableau Dashboard Generator.

Four-page flow:
    1. Upload   — drag-and-drop CSV/Excel, preview rows, see quality score
    2. Configure — set business goals, chart preferences, output format
    3. Process  — run the pipeline with a live progress bar
    4. Results  — inspect what was generated and download the .twb/.twbx
"""

import io
import os
import sys
import time
import logging
import tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st
from loguru import logger as _loguru_root

# make sure the project modules are importable
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from scripts.schemas import WorkflowConfig
from scripts.workflow import DashboardGeneratorWorkflow
from scripts.logger_config import get_logger

logger = get_logger("app")

#--- logging setup ---------------

LOG_BUFFER = io.StringIO()

def _setup_logging():
    """Wire up a stream handler that writes into LOG_BUFFER so
    the UI can show log lines to the user without touching stdout."""
    root = logging.getLogger()
    # avoid adding duplicate handlers on every rerun
    for h in root.handlers[:]:
        if getattr(h, "_streamlit_log_handler", False):
            root.removeHandler(h)

    handler = logging.StreamHandler(LOG_BUFFER)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    handler._streamlit_log_handler = True
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # also pipe loguru output into the same buffer so every module
    # that uses loguru still shows up in the Streamlit log panel
    _loguru_root.add(
        LOG_BUFFER,
        format="{time:HH:mm:ss}  [{level}]  {extra[module_name]} — {message}",
        level="INFO",
        filter=lambda record: "module_name" in record["extra"],
    )

_setup_logging()

# --- constants --------------

ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".xls")
MAX_FILE_SIZE_MB = 200

PAGES = ["upload", "configure", "process", "results"]

PAGE_LABELS = {
    "upload":    "1 · Upload",
    "configure": "2 · Configure",
    "process":   "3 · Process",
    "results":   "4 · Results",
}

# -- session state helpers --------------

def _init_state():
    """Set defaults on first load."""
    defaults = {
        "page": "upload",
        "uploaded_file_path": None,
        "uploaded_file_name": None,
        "df_preview": None,
        "quality_quick": None,
        "config": {
            "provider": "gemini",
            "model_name": "gemini-3-flash-preview",
            "api_key": "",
            "endpoint_url": "",
        },
        "pipeline_result": None,
        "pipeline_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _go(page):
    st.session_state.page = page

# -- page layout boilerplate ------------------

def _page_header():
    """Render the top navigation strip and title."""
    st.set_page_config(
        page_title="Tableau Dashboard Generator",
        page_icon="",
        layout="wide",
    )

    st.markdown(
        "<h2 style='margin-bottom:0'>AI Tableau Dashboard Generator</h2>"
        "<p style='color:grey; margin-top:0'>Upload data → Configure → Generate → Download</p>",
        unsafe_allow_html=True,
    )

    # breadcrumb-style nav
    cols = st.columns(len(PAGES))
    for i, page_key in enumerate(PAGES):
        label = PAGE_LABELS[page_key]
        is_current = (st.session_state.page == page_key)
        if is_current:
            cols[i].markdown(f"**▸ {label}**")
        else:
            cols[i].markdown(f"  {label}")

    st.divider()

#  PAGE 1 — UPLOAD
# --------------------------------------

def _validate_uploaded_file(uploaded):
    """Check extension and size.  Returns (ok, error_msg)."""
    name = uploaded.name
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type **{ext}**. Please upload a CSV or Excel file."

    size_mb = uploaded.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"File is {size_mb:.1f} MB — max allowed is {MAX_FILE_SIZE_MB} MB."

    return True, ""


def _load_preview(path: str, nrows=500) -> pd.DataFrame:
    """Quick load of the first N rows for the preview table."""
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, nrows=nrows)
    else:
        return pd.read_excel(path, nrows=nrows)


def _quick_quality(df: pd.DataFrame) -> dict:
    """Lightweight quality stats computed purely in pandas —
    no pipeline dependency so the upload page stays fast."""
    total_cells = df.shape[0] * df.shape[1]
    missing = int(df.isnull().sum().sum())
    dupes = int(df.duplicated().sum())

    missing_pct = (missing / total_cells * 100) if total_cells else 0
    dupe_pct = (dupes / len(df) * 100) if len(df) else 0

    # rough quality score (same formula as DataProcessor.assess_quality)
    score = 100.0
    score -= missing_pct * 1.5
    score -= dupe_pct * 0.5
    score = max(0.0, min(100.0, score))

    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "missing": missing,
        "missing_pct": round(missing_pct, 2),
        "dupes": dupes,
        "dupe_pct": round(dupe_pct, 2),
        "score": round(score, 1),
    }


def page_upload():
    st.subheader("Upload your dataset")

    uploaded = st.file_uploader(
        "Drag a CSV or Excel file here",
        type=["csv", "xlsx", "xls"],
        help="Max 200 MB.  The file will be saved to a temp directory for processing.",
    )

    if uploaded is not None:
        ok, err = _validate_uploaded_file(uploaded)
        if not ok:
            st.error(err)
            return

        # save to a temp location so the pipeline can read it later
        tmp_dir = tempfile.mkdtemp(prefix="tableau_gen_")
        dest = os.path.join(tmp_dir, uploaded.name)
        with open(dest, "wb") as f:
            f.write(uploaded.getbuffer())

        st.session_state.uploaded_file_path = dest
        st.session_state.uploaded_file_name = uploaded.name

        # load a preview
        try:
            df = _load_preview(dest)
            st.session_state.df_preview = df
            st.session_state.quality_quick = _quick_quality(df)
        except Exception as exc:
            st.error(f"Could not parse file: {exc}")
            return

    # show preview if we have one
    if st.session_state.df_preview is not None:
        df = st.session_state.df_preview
        q = st.session_state.quality_quick

        st.success(f"**{st.session_state.uploaded_file_name}** loaded successfully.")

        # quality metrics row
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{q['rows']:,}")
        c2.metric("Columns", q["cols"])
        c3.metric("Missing cells", f"{q['missing']:,}", delta=f"{q['missing_pct']}%", delta_color="inverse")

        # data preview
        st.markdown("##### Data preview (first rows)")
        st.dataframe(df.head(20), use_container_width=True, height=320)

        # column type breakdown
        with st.expander("Column details"):
            col_info = []
            for col in df.columns:
                dtype = str(df[col].dtype)
                n_missing = int(df[col].isnull().sum())
                n_unique = int(df[col].nunique())
                col_info.append({
                    "Column": col,
                    "Dtype": dtype,
                    "Missing": n_missing,
                    "Unique values": n_unique,
                })
            st.dataframe(pd.DataFrame(col_info), use_container_width=True, hide_index=True)

        # duplicate info
        if q["dupes"] > 0:
            st.warning(f"Found {q['dupes']} duplicate rows ({q['dupe_pct']}%).")

        st.button("Next → Configure", on_click=_go, args=("configure",), type="primary")

    else:
        st.info("Upload a file to get started.")

#  PAGE 2 — CONFIGURE
# ---------------------------------------

def page_configure():
    if st.session_state.uploaded_file_path is None:
        st.warning("No file uploaded yet.")
        st.button("← Back to Upload", on_click=_go, args=("upload",))
        return

    st.subheader("Configure generation settings")
    st.caption(f"File: **{st.session_state.uploaded_file_name}**")

    cfg = st.session_state.config

    st.markdown("LLM Configuration")
    st.markdown(
        "Enable AI-driven analysis by providing your API details. "
        "If left blank, the app will smoothly default to rule-based heuristics."
    )

    col_left, col_right = st.columns(2, gap="large")

    with col_left:
        cfg["provider"] = st.selectbox(
            "Provider",
            options=["gemini", "azure"],
            index=["gemini", "azure"].index(cfg.get("provider", "gemini")),
        )

        cfg["model_name"] = st.text_input(
            "Model Name",
            value=cfg.get("model_name", "gemini-3-flash-preview"),
        )

    with col_right:
        cfg["api_key"] = st.text_input(
            "API Key",
            value=cfg.get("api_key", ""),
            type="password",
            help="Your provider API key. Stays in your browser session only.",
        )

        cfg["endpoint_url"] = st.text_input(
            "Endpoint URL",
            value=cfg.get("endpoint_url", ""),
            help="Required for Azure OpenAI. Leave blank for Gemini.",
        )

    st.session_state.config = cfg

    st.divider()

    c1, _, c2 = st.columns([1, 4, 1])
    c1.button("← Upload", on_click=_go, args=("upload",))
    c2.button("Run pipeline →", on_click=_go, args=("process",), type="primary")

#  PAGE 3 — PROCESS
# ---------------------------------------

# friendly labels for each pipeline stage
STAGE_LABELS = {
    "pending":      "Waiting to start…",
    "validating":   "Validating data…",
    "profiling":    "Profiling columns…",
    "analyzing":    "Running AI analysis…",
    "recommending": "Choosing visualizations…",
    "generating":   "Building Tableau workbook…",
    "finalizing":   "Wrapping up…",
    "completed":    "Done!",
    "failed":       "Pipeline failed.",
}


def page_process():
    if st.session_state.uploaded_file_path is None:
        st.warning("No file uploaded.")
        st.button("← Upload", on_click=_go, args=("upload",))
        return

    st.subheader("Generating your dashboard")

    # if we already have a result, just show it and let the user move on
    if st.session_state.pipeline_result is not None:
        result = st.session_state.pipeline_result
        stage = result.get("progress", {}).get("current_stage", "unknown")

        if stage == "completed":
            st.success("Pipeline finished successfully.")
        elif stage == "failed":
            st.error("Pipeline ended with errors (see below).")
        else:
            st.info(f"Pipeline status: {stage}")

        _show_errors(result)

        c1, _, c2 = st.columns([1, 4, 1])
        c1.button("← Configure", on_click=_go, args=("configure",))
        c2.button("View results →", on_click=_go, args=("results",), type="primary")

        if st.button("Re-run pipeline"):
            st.session_state.pipeline_result = None
            st.rerun()
        return

    # -- run the pipeline --

    file_path = st.session_state.uploaded_file_path
    cfg_dict = st.session_state.config

    wf_config = WorkflowConfig(
        output_format=cfg_dict.get("output_format", "twb"),
        use_ai_analysis=bool(cfg_dict.get("api_key")),
    )

    # pass LLM credentials through as extra fields — WorkflowConfig
    # has extra="allow" so these get carried into the analyze node
    if cfg_dict.get("api_key"):
        wf_config.llm_api_key = cfg_dict["api_key"]
        wf_config.llm_provider = cfg_dict.get("provider", "gemini")
        wf_config.llm_model = cfg_dict.get("model_name", "")
        if cfg_dict.get("endpoint_url"):
            wf_config.llm_endpoint = cfg_dict["endpoint_url"]

    progress_bar = st.progress(0, text="Starting pipeline…")
    status_text = st.empty()
    log_expander = st.expander("Pipeline logs", expanded=False)

    # this dict collects the latest state from the callback
    tracker = {"last_pct": 0.0, "last_msg": ""}

    def on_progress(node_name, state):
        prog = state.get("progress", {})
        pct = prog.get("percent_complete", 0)
        stage = prog.get("current_stage", "pending")
        msg = prog.get("message", "")

        tracker["last_pct"] = pct
        tracker["last_msg"] = msg

        label = STAGE_LABELS.get(stage, stage)
        clamped = min(int(pct), 100)
        progress_bar.progress(clamped, text=f"{label}  ({clamped}%)")
        status_text.caption(msg)

    # actually run it
    wf = DashboardGeneratorWorkflow()

    try:
        result = wf.run_step_by_step(
            file_path=file_path,
            config=wf_config,
            on_progress=on_progress,
        )
    except Exception as exc:
        st.error(f"Pipeline crashed: {exc}")
        logger.exception("Pipeline crashed")
        result = {
            "progress": {"current_stage": "failed", "percent_complete": 0, "message": str(exc)},
            "errors": [{"stage": "unknown", "message": str(exc), "recoverable": False}],
        }

    # finish up the progress bar
    final_stage = result.get("progress", {}).get("current_stage", "unknown")
    if final_stage == "completed":
        progress_bar.progress(100, text="Done!")
    elif final_stage == "failed":
        progress_bar.progress(100, text="Failed — check errors below.")
    else:
        progress_bar.progress(100, text=f"Finished (stage: {final_stage})")

    st.session_state.pipeline_result = result

    # dump the captured log lines
    LOG_BUFFER.seek(0)
    log_content = LOG_BUFFER.read()
    if log_content.strip():
        with log_expander:
            st.code(log_content, language="text")

    _show_errors(result)

    if final_stage == "completed":
        st.success("Completed. Next go to the Results page.")
    elif final_stage == "failed":
        st.error("The pipeline hit errors.  You may want to adjust your config and retry.")

    c1, _, c2 = st.columns([1, 4, 1])
    c1.button("← Configure", on_click=_go, args=("configure",))
    c2.button("View results →", on_click=_go, args=("results",), type="primary")


def _show_errors(result):
    errors = result.get("errors", [])
    if not errors:
        return

    with st.expander(f"⚠ {len(errors)} error(s) / warning(s)", expanded=True):
        for e in errors:
            stage = e.get("stage", "?")
            msg = e.get("message", "")
            recoverable = e.get("recoverable", True)

            if recoverable:
                st.warning(f"**[{stage}]** {msg}")
            else:
                st.error(f"**[{stage}]** {msg}")


#  PAGE 4 — RESULTS
# --------------------------------------

def page_results():
    result = st.session_state.pipeline_result

    if result is None:
        st.warning("No results yet — run the pipeline first.")
        st.button("← Go to Upload", on_click=_go, args=("upload",))
        return

    st.subheader("Results")

    final_stage = result.get("progress", {}).get("current_stage", "unknown")
    if final_stage != "completed":
        st.warning(f"Pipeline ended in stage **{final_stage}** — results may be partial.")

    # --- download section --------------------

    output_path = result.get("output_path")
    if output_path and os.path.isfile(output_path):
        st.markdown("##### Generated workbook")

        # the pipeline always writes both formats when possible.
        # output_path points to the .twb — derive the .twbx path from it.
        twb_path = output_path
        twbx_path = output_path.replace(".twb", ".twbx")
        # make sure we don't match a .twbx that was already the output_path
        if twbx_path == twb_path:
            twbx_path = None

        has_twb = os.path.isfile(twb_path)
        has_twbx = twbx_path and os.path.isfile(twbx_path)

        if has_twb and has_twbx:
            col_a, col_b = st.columns(2)
            with col_a:
                twb_name = os.path.basename(twb_path)
                twb_size = os.path.getsize(twb_path) / 1024
                st.markdown(f"**{twb_name}**  ({twb_size:.1f} KB)")
                with open(twb_path, "rb") as f:
                    st.download_button(
                        label=f"Download .twb",
                        data=f.read(),
                        file_name=twb_name,
                        mime="application/octet-stream",
                    )
            with col_b:
                twbx_name = os.path.basename(twbx_path)
                twbx_size = os.path.getsize(twbx_path) / 1024
                st.markdown(f"**{twbx_name}**  ({twbx_size:.1f} KB)")
                with open(twbx_path, "rb") as f:
                    st.download_button(
                        label=f"Download .twbx",
                        data=f.read(),
                        file_name=twbx_name,
                        mime="application/octet-stream",
                    )
        else:
            # only one file available — just show that one
            file_name = os.path.basename(output_path)
            file_size = os.path.getsize(output_path) / 1024
            st.markdown(f"**{file_name}**  ({file_size:.1f} KB)")
            with open(output_path, "rb") as f:
                st.download_button(
                    label=f"Download {file_name}",
                    data=f.read(),
                    file_name=file_name,
                    mime="application/octet-stream",
                    type="primary",
                )
    else:
        st.info("No output file was produced.  Check the Process page for errors.")

    # -- dataset overview ---------------------------

    ds = result.get("dataset_schema")
    qr = result.get("quality_report")

    if ds:
        st.markdown("##### Dataset summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{ds.get('row_count', 0):,}")
        m2.metric("Columns", ds.get("column_count", 0))

        if qr:
            score = qr.get("quality_score", "n/a")
            if isinstance(score, (int, float)):
                score = round(score)
            m3.metric("Quality", f"{score}")
            dup_count = qr.get("duplicates", {}).get("duplicate_row_count", 0)
            m4.metric("Duplicate rows", dup_count)

        with st.expander("Column breakdown"):
            col_data = []
            for c in ds.get("columns", []):
                col_data.append({
                    "Name": c.get("name"),
                    "Type": c.get("semantic_type"),
                    "Missing": c.get("missing_count", 0),
                    "Missing %": f"{c.get('missing_ratio', 0) * 100:.1f}%",
                    "Unique": c.get("unique_count", ""),
                })
            if col_data:
                st.dataframe(pd.DataFrame(col_data), use_container_width=True, hide_index=True)

    # -- errors ------------

    _show_errors(result)

    # -- logs ----------------------------
    LOG_BUFFER.seek(0)
    log_text = LOG_BUFFER.read()
    if log_text.strip():
        with st.expander("Full pipeline logs"):
            st.code(log_text, language="text")

    # nav
    st.divider()
    c1, c2, _ = st.columns([1, 1, 4])
    c1.button("← Back to Process", on_click=_go, args=("process",))
    c2.button("Start over", on_click=_reset)


def _reset():
    """Clear everything and go back to upload."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # _init_state will re-populate defaults on next run


#  MAIN ROUTER
# --------------------------------------

def main():
    _init_state()
    _page_header()

    page = st.session_state.page

    if page == "upload":
        page_upload()
    elif page == "configure":
        page_configure()
    elif page == "process":
        page_process()
    elif page == "results":
        page_results()
    else:
        st.error(f"Unknown page: {page}")
        _go("upload")


if __name__ == "__main__":
    main()
