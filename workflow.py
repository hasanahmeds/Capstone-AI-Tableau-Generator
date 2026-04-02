"""
DashboardGeneratorWorkflow
--------------------------
Orchestrates the end-to-end dashboard generation pipeline using LangGraph.

The pipeline chains six stages together:

    validate  ->  profile  ->  analyze  ->  recommend  ->  generate  ->  finalize

Each stage is implemented as a standalone node function that reads from the
shared state dict, does its work, and returns only the keys it wants to
update.  LangGraph takes care of merging partial updates back into the
full state between nodes.

Module dependencies (all project-local):
    data_processor.py           ->  DataProcessor (load / validate / quality / profile)
    dashboard_analyzer.py       ->  DashboardAnalyzer (LLM + rule-based analysis)
    tableau_workbook_generator.py -> TableauWorkbookGenerator (XML / TWBX export)
    schemas.py                  ->  every Pydantic model the pipeline touches
    error_handling.py           ->  retry policies, resilient I/O helpers

Usage:
    from workflow import DashboardGeneratorWorkflow

    wf = DashboardGeneratorWorkflow()
    result = wf.run("sales_2024.csv")

    print(result["progress"]["current_stage"])   # "completed"
    print(result["output_path"])                 # "output/sales_2024_dashboard.twb"
"""

import io
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from uuid import uuid4

import pandas as pd
from langgraph.graph import StateGraph, END, START

# project-local imports
from schemas import (
    WorkflowState,
    WorkflowConfig,
    WorkflowProgress,
    WorkflowError,
    WorkflowStage,
    AnalysisResult,
    KPIRecommendation,
    DatasetSchema,
    QualityReport,
    ProfileReport,
    DashboardSpec,
    WorkbookSpec,
    WorksheetSpec,
    VisualizationSpec,
    ChartType,
    build_dashboard_spec,
)


# Logger — can attach their own handlers (JSON, file, etc.) to the
# "workflow" logger or the root logger.

logger = logging.getLogger("workflow")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(name)s — %(message)s")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# LangGraph needs a TypedDict (not a plain dict) to know which keys exist
# across the entire pipeline.  Without this definition the merge logic only
# keeps the keys returned by the LAST node, and everything upstream vanishes.

class GraphState(TypedDict, total=False):
    """Typed shape of the state dictionary that travels through the graph."""
    run_id: str
    file_path: Optional[str]
    config: Dict[str, Any]

    # serialised DataFrame (orient="split") so we don't have to re-read
    # the file every time a downstream node needs the raw data
    dataframe_json: Optional[str]

    # structured outputs from each pipeline stage
    dataset_schema: Optional[Dict[str, Any]]
    quality_report: Optional[Dict[str, Any]]
    profile_report: Optional[Dict[str, Any]]
    analysis_result: Optional[Dict[str, Any]]
    dashboard_spec: Optional[Dict[str, Any]]
    workbook_spec: Optional[Dict[str, Any]]
    output_path: Optional[str]

    # progress + errors are always present
    progress: Dict[str, Any]
    errors: List[Dict[str, Any]]

# helpers that keep the node functions tidy

def _progress(stage: WorkflowStage, pct: float, msg: str) -> dict:
    """Snapshot the pipeline's current progress as a serialisable dict."""
    return WorkflowProgress(
        current_stage=stage,
        percent_complete=round(pct, 1),
        message=msg,
    ).model_dump()


def _error(
    stage: WorkflowStage,
    msg: str,
    recoverable: bool = True,
    details: Optional[Dict[str, Any]] = None,
) -> dict:
    """Build a serialisable error record."""
    return WorkflowError(
        stage=stage,
        message=msg,
        recoverable=recoverable,
        details=details or {},
    ).model_dump()


def _rebuild_df(state: dict) -> pd.DataFrame:
    """Reconstruct the DataFrame from its JSON-serialised form in state."""
    raw = state.get("dataframe_json")
    if not raw:
        return pd.DataFrame()
    return pd.read_json(io.StringIO(raw), orient="split")


def _rebuild_model(raw, model_cls):
    """Turn a dict (or already-instantiated model) back into its Pydantic class."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return model_cls(**raw)
    return raw   # already the right type

# NODE 1 — VALIDATE

def validate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Load the raw file, detect column types, and run quality checks.

    This node wraps DataProcessor.load(), .validate(), and .assess_quality()
    from data_processor.py.  On any failure (file not found, empty data,
    parse error) it records a non-recoverable error and the conditional
    router after this node will skip the rest of the pipeline.
    """
    logger.info("validate_node: starting data validation")

    file_path = state.get("file_path")
    config_raw = state.get("config", {})
    config = _rebuild_model(config_raw, WorkflowConfig)
    errors = list(state.get("errors", []))

    # no file? nothing to do.
    if not file_path:
        errors.append(_error("validating", "No file_path provided.", recoverable=False))
        return {
            "progress": _progress("failed", 0.0, "Missing file path."),
            "errors": errors,
        }

    # pull in the real DataProcessor from the project
    from data_processor import DataProcessor, DataLoadError, DataValidationError
    processor = DataProcessor()

    # --- load the file ---
    try:
        df = processor.load(file_path)
    except (DataLoadError, Exception) as exc:
        logger.error("validate_node: file load failed — %s", exc)
        errors.append(
            _error("validating", f"File load error: {exc}", recoverable=False)
        )
        return {
            "progress": _progress("failed", 0.0, str(exc)),
            "errors": errors,
        }

    # --- detect column types + build the DatasetSchema ---
    try:
        filename = os.path.basename(file_path)
        dataset_schema = processor.validate(
            df, source_filename=filename, dataset_name=filename,
        )
    except (DataValidationError, Exception) as exc:
        logger.error("validate_node: validation failed — %s", exc)
        errors.append(
            _error("validating", f"Validation error: {exc}", recoverable=False)
        )
        return {
            "progress": _progress("failed", 5.0, str(exc)),
            "errors": errors,
        }

    # --- quality assessment ---
    quality_report = processor.assess_quality(df, dataset_schema)

    score = quality_report.quality_score or 0.0
    logger.info(
        "validate_node: rows=%d  cols=%d  quality=%.1f",
        dataset_schema.row_count, dataset_schema.column_count, score,
    )

    # change the DataFrame to JSON so downstream nodes can get it
    # without touching the filesystem again
    df_json = df.to_json(orient="split", date_format="iso")

    return {
        "dataframe_json": df_json,
        "dataset_schema": dataset_schema.model_dump(),
        "quality_report": quality_report.model_dump(),
        "progress": _progress(
            "validating", 20.0,
            f"Validated {dataset_schema.row_count} rows, quality score {score}.",
        ),
        "errors": errors,
    }

# NODE 2 — PROFILE

def profile_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Compute per-column descriptive statistics.

    Uses DataProcessor.profile() to build a ProfileReport.  The profile
    feeds into the AI analysis node and also into the rule-based fallback
    for visualization recommendations.
    """
    logger.info("profile_node: building column profiles")

    from data_processor import DataProcessor
    processor = DataProcessor()

    dataset_schema = _rebuild_model(state.get("dataset_schema"), DatasetSchema)
    df = _rebuild_df(state)

    profile_report = processor.profile(df, dataset_schema)

    logger.info(
        "profile_node: profiled %d columns", len(profile_report.column_profiles)
    )

    return {
        "profile_report": profile_report.model_dump(),
        "progress": _progress(
            "profiling", 35.0,
            f"Profiled {len(profile_report.column_profiles)} columns.",
        ),
    }

# NODE 3 — ANALYZE (AI / rule-based)

def analyze_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Run DashboardAnalyzer.analyze() + recommend_kpis().

    When an API key is present the analyzer tries the LLM first.  On
    failure (network, quota, bad JSON) it falls back to deterministic
    automatically — the fallback logic is inside
    DashboardAnalyzer itself so we don't duplicate it here.

    We translate the analyzer's raw dict output into the AnalysisResult
    Pydantic model so the rest of the pipeline has a consistent schema.
    """
    logger.info("analyze_node: running analysis")

    config = _rebuild_model(state.get("config"), WorkflowConfig)
    dataset_schema = _rebuild_model(state.get("dataset_schema"), DatasetSchema)
    df = _rebuild_df(state)
    errors = list(state.get("errors", []))

    from dashboard_analyzer import DashboardAnalyzer

    # Build the analyzer.  When use_ai_analysis is False we explicitly
    # skip passing an api_key so the analyzer goes straight to its
    # rule-based fallbacks without even attempting an LLM call.
    analyzer_kwargs = {}
    if config.use_ai_analysis:
        # the caller can stuff LLM credentials into the config's extra fields
        # (WorkflowConfig has extra="allow" so arbitrary keys are accepted)
        api_key = getattr(config, "llm_api_key", None)
        endpoint = getattr(config, "llm_endpoint", None)
        model = getattr(config, "llm_model", None)
        provider = getattr(config, "llm_provider", None)

        if api_key:
            analyzer_kwargs["api_key"] = api_key
        if endpoint:
            analyzer_kwargs["endpoint"] = endpoint
        if model:
            analyzer_kwargs["model"] = model
        if provider:
            analyzer_kwargs["provider"] = provider

    analyzer = DashboardAnalyzer(**analyzer_kwargs)

    # Hand the already-loaded DataFrame to the analyzer directly
    # instead of making it read the file from disk a second time.
    # DashboardAnalyzer stores the df on self.df and every method
    # checks self.df before doing anything.
    analyzer.df = df

    # --- run the two core methods ---
    overview = {}
    kpi_result = {}
    try:
        overview = analyzer.analyze()
        kpi_result = analyzer.recommend_kpis()
    except Exception as exc:
        logger.warning("analyze_node: analyzer raised — %s", exc)
        errors.append(_error(
            "analyzing",
            f"Analysis error (non-fatal): {exc}",
            recoverable=True,
            details={"traceback": traceback.format_exc()},
        ))

    # --- translate analyzer output → AnalysisResult schema ---------------
    kpis = _extract_kpi_list(kpi_result, dataset_schema)
    chart_hints = _extract_chart_hints(overview, dataset_schema)

    analysis_result = AnalysisResult(
        dataset_id=dataset_schema.dataset_id,
        summary=overview.get("dataset_description", ""),
        kpis=kpis,
        suggested_chart_types=chart_hints,
        business_domain=overview.get("dataset_domain"),
        extra_metadata={
            "column_roles": overview.get("column_roles", {}),
            "time_coverage": overview.get("time_coverage", {}),
            "key_entities": overview.get("key_entities", []),
            "raw_kpis": kpi_result,
        },
    )

    logger.info(
        "analyze_node: found %d KPIs, domain=%s",
        len(kpis), analysis_result.business_domain,
    )

    return {
        "analysis_result": analysis_result.model_dump(),
        "progress": _progress(
            "analyzing", 50.0,
            f"Identified {len(kpis)} KPIs.",
        ),
        "errors": errors,
    }

# NODE 4 — RECOMMEND VISUALIZATIONS

def recommend_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Pick chart types and produce a DashboardSpec.

    We start with the build_dashboard_spec() heuristic from schemas.py
    as a baseline (column-type-based: datetime→line, categorical→bar, etc.)
    then layer on any additional charts that the dashboard_analyzer's KPI
    recommendations suggest.  This way the trend_metrics and
    comparative_metrics from _fallback_kpis() / the LLM actually show
    up in the final workbook instead of being silently dropped.
    """
    logger.info("recommend_node: selecting visualizations")

    dataset_schema = _rebuild_model(state.get("dataset_schema"), DatasetSchema)
    df = _rebuild_df(state)
    config = _rebuild_model(state.get("config"), WorkflowConfig)
    analysis_raw = state.get("analysis_result", {})

    # --- baseline charts from column-type heuristics ---
    dashboard_spec = build_dashboard_spec(dataset_schema, df)

    # keep track of which (x, y) combos we already have so we
    # don't create duplicate charts for the same pair of columns
    existing_pairs = set()
    for v in dashboard_spec.visuals:
        existing_pairs.add((v.x, v.y))

    known_cols = {c.name for c in dataset_schema.columns}

    # --- pull the raw KPI dict that analyze_node stashed in extra_metadata ---
    raw_kpis = {}
    if isinstance(analysis_raw, dict):
        raw_kpis = analysis_raw.get("extra_metadata", {}).get("raw_kpis", {})

    # trend_metrics → line charts
    # each entry looks like: {name, formula, time_grain, source_columns: [time_col, measure]}
    for trend in raw_kpis.get("trend_metrics", []):
        src = trend.get("source_columns", [])
        if len(src) < 2:
            continue
        time_col, measure_col = src[0], src[1]

        # skip if the columns don't exist or we already have this combo
        if time_col not in known_cols or measure_col not in known_cols:
            continue
        if (time_col, measure_col) in existing_pairs:
            continue

        agg = _parse_agg_from_formula(trend.get("formula", ""))
        dashboard_spec.visuals.append(VisualizationSpec(
            chart_type="line",
            title=trend.get("name", f"{measure_col} over {time_col}"),
            x=time_col,
            y=measure_col,
            aggregation=agg,
            rationale=f"Trend metric from AI analysis.",
            confidence=0.75,
        ))
        existing_pairs.add((time_col, measure_col))

    # comparative_metrics → bar charts
    # each entry looks like: {name, measure, compare_by, source_columns: [dim, measure]}
    for comp in raw_kpis.get("comparative_metrics", []):
        src = comp.get("source_columns", [])
        if len(src) < 2:
            continue
        dim_col, measure_col = src[0], src[1]

        if dim_col not in known_cols or measure_col not in known_cols:
            continue
        if (dim_col, measure_col) in existing_pairs:
            continue

        agg = _parse_agg_from_formula(comp.get("measure", ""))
        dashboard_spec.visuals.append(VisualizationSpec(
            chart_type="bar",
            title=comp.get("name", f"{measure_col} by {dim_col}"),
            x=dim_col,
            y=measure_col,
            aggregation=agg,
            rationale=f"Comparative metric from AI analysis.",
            confidence=0.72,
        ))
        existing_pairs.add((dim_col, measure_col))

    # primary KPIs with measures that aren't charted yet → histogram
    for kpi in raw_kpis.get("primary_kpis", []):
        src = kpi.get("source_columns", [])
        if not src:
            continue
        col = src[0]
        if col not in known_cols:
            continue
        # only add if this column hasn't been used as an x axis already
        already_used = any(col == pair[0] for pair in existing_pairs)
        if already_used:
            continue

        # check column is actually numeric before making a histogram
        if col not in dataset_schema.numeric_columns:
            continue

        dashboard_spec.visuals.append(VisualizationSpec(
            chart_type="histogram",
            title=f"Distribution of {col}",
            x=col,
            rationale=f"Primary KPI column — showing its distribution.",
            confidence=0.65,
        ))
        existing_pairs.add((col, None))

    # respect the user's upper limit on chart count
    if len(dashboard_spec.visuals) > config.max_visualizations:
        dashboard_spec.visuals = dashboard_spec.visuals[:config.max_visualizations]

    logger.info(
        "recommend_node: recommended %d charts (baseline + %d from analyzer KPIs)",
        len(dashboard_spec.visuals),
        max(0, len(dashboard_spec.visuals) - 4),  # rough count of extras
    )

    return {
        "dashboard_spec": dashboard_spec.model_dump(),
        "progress": _progress(
            "recommending", 65.0,
            f"Recommended {len(dashboard_spec.visuals)} charts.",
        ),
    }


def _parse_agg_from_formula(formula: str) -> str:
    """Pull the aggregation keyword out of a formula like 'SUM([Sales])'."""
    if not formula:
        return "sum"
    upper = formula.upper()
    for keyword in ("SUM", "AVG", "COUNT", "MIN", "MAX", "MEDIAN"):
        if keyword in upper:
            return keyword.lower()
    return "sum"

# NODE 5 — GENERATE TABLEAU WORKBOOK

def generate_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Turn the DashboardSpec into a .twb or .twbx file on disk.

    Delegates the heavy lifting to TableauWorkbookGenerator.  The
    generator's constructor accepts a DashboardSpec, a DatasetSchema,
    and the raw DataFrame, then _plan_worksheets() -> _build_xml() ->
    export_twb()/export_twbx() takes care of the rest.

    After generation we sync the dashboard_spec back into the state
    so it reflects any worksheets the generator injected internally
    (e.g. time-series fallback charts for date columns the schema missed).
    """
    logger.info("generate_node: building workbook")

    dataset_schema = _rebuild_model(state.get("dataset_schema"), DatasetSchema)
    dashboard_spec = _rebuild_model(state.get("dashboard_spec"), DashboardSpec)
    df = _rebuild_df(state)
    config = _rebuild_model(state.get("config"), WorkflowConfig)
    errors = list(state.get("errors", []))

    from tableau_workbook_generator import TableauWorkbookGenerator

    output_path = None
    workbook_spec = None
    updated_dashboard_spec = None

    try:
        generator = TableauWorkbookGenerator(
            dashboard_spec=dashboard_spec,
            dataset_schema=dataset_schema,
            dataframe=df,
        )

        # run the generator's own validation before writing anything
        is_valid, issues = generator.validate()
        if issues:
            for issue in issues:
                logger.warning("generate_node: validation issue — %s", issue)

        # decide the output filename from the source dataset name
        stem = Path(state.get("file_path", "dashboard")).stem
        out_dir = "output"

        # always generate the .twb first since it's the plain XML version
        twb_file = os.path.join(out_dir, f"{stem}_dashboard.twb")
        output_path = generator.export_twb(twb_file)
        logger.info("generate_node: .twb written to %s", output_path)

        # try generating the .twbx too — this bundles the data file
        # inside a zip so the workbook is self-contained and portable.
        # it needs pantab for the hyper extract, so if that's not
        # installed we just log a warning and move on with the .twb
        try:
            twbx_file = os.path.join(out_dir, f"{stem}_dashboard.twbx")
            twbx_path = generator.export_twbx(
                twbx_file, data_file=state.get("file_path"),
            )
            logger.info("generate_node: .twbx written to %s", twbx_path)
        except Exception as twbx_exc:
            logger.warning("generate_node: .twbx generation skipped — %s", twbx_exc)

        # grab the WorkbookSpec the generator built internally so we
        # can pass it downstream (e.g. for tests and Streamlit display)
        if generator._workbook_spec is not None:
            workbook_spec = generator._workbook_spec.model_dump()

        # --- sync dashboard_spec with what actually ended up in the file ---
        # The generator's _plan_worksheets() can inject extra worksheets
        # (time-series fallback for date columns the schema missed).
        # Those worksheets are in workbook_spec but not in the original
        # dashboard_spec.  We build VisualizationSpec entries for them
        # so the pipeline state reflects the actual contents of the .twb.
        if generator._workbook_spec is not None:
            ws_titles_in_spec = {v.title for v in dashboard_spec.visuals}
            injected = []
            for ws in generator._workbook_spec.worksheets:
                if ws.name not in ws_titles_in_spec:
                    # this worksheet was added by the generator internally
                    mark_to_chart = {
                        "Bar": "bar", "Line": "line", "Circle": "scatter",
                        "Square": "heatmap", "Pie": "pie", "Text": "table",
                    }
                    chart_type = mark_to_chart.get(ws.mark_type, "bar")
                    x_col = ws.columns_shelf[0] if ws.columns_shelf else None
                    y_col = ws.rows_shelf[0] if ws.rows_shelf else None
                    agg = ws.aggregation.lower() if ws.aggregation else None

                    injected.append(VisualizationSpec(
                        chart_type=chart_type,
                        title=ws.name,
                        x=x_col,
                        y=y_col,
                        aggregation=agg,
                        rationale="Auto-injected by workbook generator (date column fallback).",
                        confidence=0.7,
                    ))

            if injected:
                logger.info(
                    "generate_node: syncing %d injected worksheets back into dashboard_spec",
                    len(injected),
                )
                dashboard_spec.visuals.extend(injected)

            updated_dashboard_spec = dashboard_spec.model_dump()

        logger.info("generate_node: workbook written to %s", output_path)

    except Exception as exc:
        logger.error("generate_node: generation failed — %s", exc)
        errors.append(_error(
            "generating",
            f"Workbook generation error: {exc}",
            recoverable=True,
            details={"traceback": traceback.format_exc()},
        ))

    result = {
        "workbook_spec": workbook_spec,
        "output_path": output_path,
        "progress": _progress(
            "generating", 85.0,
            "Workbook spec ready." if output_path is None
            else f"Workbook saved to {output_path}.",
        ),
        "errors": errors,
    }

    # only overwrite dashboard_spec if we actually updated it — don't
    # clobber the existing value on failure
    if updated_dashboard_spec is not None:
        result["dashboard_spec"] = updated_dashboard_spec

    return result

# NODE 6 — FINALIZE

def finalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap-up: stamp the finish time and decide whether the run succeeded.

    Checks all accumulated errors.  If any are non-recoverable the final
    status is "failed"; otherwise "completed".  This gives the Streamlit UI
    a single flag to check.
    """
    logger.info("finalize_node: wrapping up")

    errors = state.get("errors", [])
    has_fatal = any(
        not (e.get("recoverable", True) if isinstance(e, dict) else e.recoverable)
        for e in errors
    )

    if has_fatal:
        progress = _progress("failed", 100.0, "Pipeline finished with fatal errors.")
    else:
        progress = _progress("completed", 100.0, "Dashboard generation complete.")

    # stamp the finish time
    progress["finished_at"] = datetime.utcnow().isoformat()

    logger.info(
        "finalize_node: status=%s  errors=%d",
        progress["current_stage"], len(errors),
    )

    return {"progress": progress}

# Internal translation helpers for the analyze_node

def _extract_kpi_list(
    kpi_result: dict,
    schema: DatasetSchema,
) -> List[KPIRecommendation]:
    """Convert the DashboardAnalyzer's raw KPI dict into typed KPIRecommendation objects.

    The analyzer returns something like:
        {
            "primary_kpis":       [ {name, formula, source_columns, ...}, ... ],
            "secondary_metrics":  [ ... ],
            "trend_metrics":      [ ... ],
            "comparative_metrics":[ ... ],
        }

    We flatten all of those into a single list, mapping 'formula'
    strings like 'SUM([Sales])' back to column + aggregation.
    """
    kpis = []
    known_cols = {c.name for c in schema.columns}

    agg_keywords = {
        "SUM": "sum", "AVG": "avg", "MIN": "min",
        "MAX": "max", "COUNT": "count", "MEDIAN": "median",
    }

    # collect entries from every kpi  the analyzer produces
    raw_items = []
    for bucket in ("primary_kpis", "secondary_metrics",
                    "trend_metrics", "comparative_metrics"):
        raw_items.extend(kpi_result.get(bucket, []))

    for item in raw_items:
        name = item.get("name", "Unnamed KPI")
        formula = item.get("formula", "")
        source_cols = item.get("source_columns", [])
        rationale = item.get("business_rationale", "")

        # figure out which column this KPI is based on
        col = None
        for c in source_cols:
            if c in known_cols:
                col = c
                break

        if col is None:
            continue  # skip KPIs that reference columns we don't have

        # parse the aggregation out of the formula string
        agg = "sum"
        upper_formula = formula.upper()
        for keyword, mapped in agg_keywords.items():
            if keyword in upper_formula:
                agg = mapped
                break

        kpis.append(KPIRecommendation(
            metric_name=name,
            column=col,
            aggregation=agg,
            rationale=rationale,
        ))

    return kpis


def _extract_chart_hints(
    overview: dict,
    schema: DatasetSchema,
) -> List[ChartType]:
    """Infer useful chart types from the column roles detected by the analyzer."""
    roles = overview.get("column_roles", {})
    hints: List[ChartType] = []

    if roles.get("temporal") and roles.get("measures"):
        hints.append("line")
    if roles.get("dimensions") and roles.get("measures"):
        hints.append("bar")
    if len(roles.get("measures", [])) >= 2:
        hints.append("scatter")
    if roles.get("measures"):
        hints.append("histogram")

    return hints

# GRAPH ASSEMBLY

def _build_graph() -> StateGraph:
    """Wire up the six nodes and compile the LangGraph.

    Straight chain — every run goes through all six nodes:

        validate -> profile -> analyze -> recommend -> generate -> finalize
    """
    graph = StateGraph(GraphState)

    # register nodes
    graph.add_node("validate",  validate_node)
    graph.add_node("profile",   profile_node)
    graph.add_node("analyze",   analyze_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("generate",  generate_node)
    graph.add_node("finalize",  finalize_node)

    # straight chain from start to end
    graph.add_edge(START, "validate")
    graph.add_edge("validate",  "profile")
    graph.add_edge("profile",   "analyze")
    graph.add_edge("analyze",   "recommend")
    graph.add_edge("recommend", "generate")
    graph.add_edge("generate",  "finalize")
    graph.add_edge("finalize",  END)

    return graph

# PUBLIC CLASS

class DashboardGeneratorWorkflow:
    """High-level entry point for the dashboard generation pipeline.

    Wraps the LangGraph StateGraph so callers (the Streamlit UI, tests,
    scripts) don't need to know anything about graph construction or
    state serialisation.

    Example
    -------
    >>> wf = DashboardGeneratorWorkflow()
    >>> result = wf.run("data/sales_2024.csv")
    >>> print(result["progress"]["current_stage"])
    'completed'
    """

    def __init__(self) -> None:
        # compile the graph once; the compiled runner is reusable
        self._graph = _build_graph()
        self._app = self._graph.compile()

    # save_graph_png() — dump the pipeline graph to a PNG image

    def save_graph_png(
        self,
        output_path: str = "output/workflow_graph.png",
    ) -> Optional[str]:
        """Render the compiled LangGraph to a PNG file.

        Uses pygraphviz under the hood (requires the graphviz system
        package and the pygraphviz pip package).  If either is missing
        the method logs a warning and returns None — the pipeline keeps
        running regardless.

        Parameters
        ----------
        output_path : str
            Where to write the image.  Parent directories are created
            automatically.

        Returns
        -------
        str or None
            The absolute path to the saved PNG, or None on failure.
        """
        try:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)

            png_bytes = self._app.get_graph().draw_png()
            out.write_bytes(png_bytes)

            logger.info("Graph PNG saved to %s (%d KB)", out.resolve(), len(png_bytes) // 1024)
            return str(out.resolve())

        except Exception as exc:
            # pygraphviz or graphviz might not be installed — that is
            # perfectly fine, the pipeline shouldn't break over a missing
            # diagram dependency
            logger.warning("Could not save graph PNG: %s", exc)
            return None

    # run() — execute the whole pipeline in one shot

    def run(
        self,
        file_path: str,
        config: Optional[WorkflowConfig] = None,
    ) -> Dict[str, Any]:
        """Run the full pipeline synchronously.

        Parameters
        ----------
        file_path : str
            Path to a CSV or Excel file.
        config : WorkflowConfig, optional
            User preferences (quality threshold, chart limits, LLM creds, etc.).

        Returns
        -------
        dict
            The final pipeline state with keys like ``progress``,
            ``output_path``, ``errors``, ``dashboard_spec``, ``workbook_spec``.
        """
        if config is None:
            config = WorkflowConfig()

        # save the graph structure as a PNG before the run starts
        # so we always have a visual record of the pipeline shape
        self.save_graph_png()

        initial_state: Dict[str, Any] = {
            "run_id": str(uuid4()),
            "file_path": file_path,
            "config": config.model_dump(),
            "errors": [],
            "progress": _progress("pending", 0.0, "Pipeline starting..."),
        }

        logger.info(
            "Starting pipeline  run=%s  file=%s",
            initial_state["run_id"], file_path,
        )

        final_state = self._app.invoke(initial_state)

        logger.info(
            "Pipeline finished  run=%s  stage=%s",
            final_state.get("run_id"),
            final_state.get("progress", {}).get("current_stage"),
        )
        return final_state

    # run_step_by_step() — for Streamlit progress bars and callbacks

    def run_step_by_step(
        self,
        file_path: str,
        config: Optional[WorkflowConfig] = None,
        on_progress: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Same as run(), but fires a callback after every node.

        Parameters
        ----------
        on_progress : callable(node_name: str, state: dict), optional
            Invoked after each pipeline node finishes.  Useful for
            driving a Streamlit progress bar or writing log lines.
        """
        if config is None:
            config = WorkflowConfig()

        # save the graph structure as a PNG before streaming starts
        self.save_graph_png()

        initial_state: Dict[str, Any] = {
            "run_id": str(uuid4()),
            "file_path": file_path,
            "config": config.model_dump(),
            "errors": [],
            "progress": _progress("pending", 0.0, "Pipeline starting..."),
        }

        # LangGraph's stream() gets one dict per completed node.
        # The dict key is the node name and the value is the partial
        # state update that node returned.
        final_state = dict(initial_state)
        for step_output in self._app.stream(initial_state):
            for node_name, partial in step_output.items():
                final_state.update(partial)

                if callable(on_progress):
                    try:
                        on_progress(node_name, final_state)
                    except Exception:
                        # a broken callback must never kill the pipeline
                        logger.debug("on_progress callback raised", exc_info=True)

        return final_state

    # convenience accessors

    @property
    def graph(self) -> StateGraph:
        """Raw LangGraph StateGraph — useful for tests and introspection."""
        return self._graph

# run: python workflow.py train.csv

if __name__ == "__main__":
    import sys
    import json

    # accept the file path from the command line, default to train.csv
    file_path = sys.argv[1] if len(sys.argv) > 1 else "train.csv"

    print(f"{'=' * 60}")
    print(f"  DashboardGeneratorWorkflow")
    print(f"  Input: {file_path}")
    print(f"{'=' * 60}\n")

    wf = DashboardGeneratorWorkflow()

    # run with step-by-step progress so we can see each node complete
    def on_progress(node_name, state):
        pct = state["progress"]["percent_complete"]
        msg = state["progress"]["message"]
        print(f"  [{node_name:12s}] {pct:5.1f}%  {msg}")

    print("Running pipeline...\n")
    result = wf.run_step_by_step(file_path, on_progress=on_progress)

    # print summary
    print(f"\n{'=' * 60}")
    print(f"  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Status:       {result['progress']['current_stage']}")
    print(f"  Quality:      {result.get('quality_report', {}).get('quality_score')}")
    print(f"  Output file:  {result.get('output_path', 'N/A')}")
    print(f"  Graph PNG:    output/workflow_graph.png")

    # dataset info
    ds = result.get("dataset_schema", {})
    if ds:
        print(f"\n  Dataset: {ds.get('row_count')} rows x {ds.get('column_count')} cols")
        print(f"    numeric:     {ds.get('numeric_columns', [])}")
        print(f"    categorical: {ds.get('categorical_columns', [])}")
        print(f"    datetime:    {ds.get('datetime_columns', [])}")

    # KPIs
    ar = result.get("analysis_result", {})
    kpis = ar.get("kpis", [])
    if kpis:
        print(f"\n  KPIs ({len(kpis)}):")
        for k in kpis:
            print(f"    - {k['metric_name']} ({k['aggregation']} of {k['column']})")

    # charts
    dash = result.get("dashboard_spec", {})
    visuals = dash.get("visuals", [])
    if visuals:
        print(f"\n  Charts ({len(visuals)}):")
        for v in visuals:
            print(f"    - {v['chart_type']}: {v['title']}")

    # worksheets
    ws = result.get("workbook_spec", {})
    sheets = ws.get("worksheets", []) if ws else []
    if sheets:
        print(f"\n  Worksheets ({len(sheets)}):")
        for w in sheets:
            print(f"    - {w['name']}  mark={w['mark_type']}")

    # errors
    errors = result.get("errors", [])
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors:
            tag = "FATAL" if not e.get("recoverable", True) else "WARN"
            print(f"    [{tag}] [{e['stage']}] {e['message']}")
    else:
        print(f"\n  Errors: None")

    # save full result to JSON
    output_filename = "output/workflow_result.json"
    try:
        Path(output_filename).parent.mkdir(parents=True, exist_ok=True)
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Full result saved to {output_filename}")
    except Exception as e:
        print(f"\n  Error saving result: {e}")

    print("  DONE")

