# AI-Powered Tableau Dashboard Generator

Drop a CSV or Excel file in, get a ready-to-open Tableau workbook out. No manual chart building, no shelf dragging, no formula writing.

This project takes any tabular dataset, figures out what the data is about, picks the right KPIs and chart types, and generates valid `.twb` and `.twbx` files that open directly in Tableau Desktop. It works with or without an LLM — if you plug in an API key (Azure OpenAI or Google Gemini), the analysis gets smarter, but if you don't, it falls back to deterministic heuristics and still produces a solid dashboard.

---

## What Problem This Solves

Building a Tableau dashboard from scratch is slow. You need to understand the dataset's domain, figure out which columns matter, decide on KPIs, pick chart types that actually make sense for the data, assign fields to the right shelves, set aggregations, and wire everything together. For someone unfamiliar with a new dataset — a new team member, a stakeholder, or an analyst who just got handed an unfamiliar CSV — this process can take hours.

This tool does that entire job automatically. Upload your data, optionally provide an LLM API key, and the pipeline handles everything from column type detection to Tableau XML generation.

---

## How It Works (The Pipeline)

The whole system runs as a six-stage pipeline orchestrated by [LangGraph](https://github.com/langchain-ai/langgraph). Each stage is a standalone function that reads from a shared state dictionary, does its work, and returns only the keys it changed. LangGraph merges the partial updates back into the full state between steps.

```
validate  →  profile  →  analyze  →  recommend  →  generate  →  finalize
```

Here's what each stage actually does:

**1. Validate** — Loads the CSV or Excel file using pandas, strips whitespace from column names, and runs semantic type inference on every column. The type detector uses a priority cascade: check for booleans first, then native datetime, then numeric types, then for object/string columns it tries numeric coercion (≥80% success → numeric), datetime coercion (≥80% → datetime), and finally decides between categorical (≤50 unique values) and text (average string length > 50 characters) based on cardinality. The output is a `DatasetSchema` with per-column metadata and convenience lists grouping columns by type.

**2. Profile** — Computes detailed statistics for every column. Numeric columns get full descriptive stats (mean, std, quartiles, min, max). Categorical columns get value counts and a top-k list. Datetime columns get the date range. Text columns get average, min, and max string lengths. The profile feeds into the AI analysis and is also used by the visualization recommender.

**3. Analyze** — This is the intelligence layer. When an LLM API key is configured, the system builds a text profile of the dataset (shape, dtypes, sample values, numeric summary, top categorical values, date ranges) and sends it to the model with a structured prompt asking it to identify the business domain, data grain, key entities, and column roles. It then asks for KPI recommendations with actual Tableau formulas (like `SUM([Sales])` or `SUM([Profit]) / SUM([Sales])`). If the LLM call fails — network error, rate limit, malformed JSON, whatever — it falls back to rule-based heuristics that classify columns using keyword matching and generate KPIs from measure columns. The pipeline never blocks on API issues.

**4. Recommend** — The `VisualizationRecommender` fires five rule sets in order: time series (datetime + numeric → line chart), categorical breakdowns (categorical + numeric → bar/pie chart), distributions (numeric → histogram, with optional box plot), numeric relationships (pairs of numerics → scatter plot), and a table fallback for sparse datasets. It filters out identifier columns (Row ID, Postal Code, etc.) before recommending charts, because summing a zip code produces nonsense. Charts get confidence scores (line = 0.85, bar = 0.80, histogram = 0.70, scatter = 0.68, pie = 0.65) and if the total exceeds the cap (default 10), only the highest-confidence ones survive.

**5. Generate** — The `TableauWorkbookGenerator` converts the abstract chart specs into real Tableau XML. It plans worksheets (mapping chart types to shelf assignments — which fields go on Rows, which go on Columns, which go on Color/Size encodings), builds the full XML tree (datasource with connection metadata, column type declarations, metadata records, worksheets with pane/mark/encoding elements, dashboard layout, and window settings), and writes it to disk. It also handles a tricky edge case: if the upstream type detection missed date columns stored as DD/MM/YYYY strings, the generator scans for them and injects time-series worksheets. Both `.twb` (plain XML) and `.twbx` (ZIP archive with data + Hyper extract) formats are generated.

**6. Finalize** — Checks all accumulated errors. If any are non-recoverable, the run status is "failed"; otherwise "completed". Dumps intermediate results as JSON to the logs directory for debugging.

---

## File Structure

```
project/
├── app.py                              # Streamlit web interface (4 pages)
├── scripts/
│   ├── __init__.py                     # Makes scripts/ a Python package
│   ├── schemas.py                      # Every Pydantic model (~920 lines)
│   ├── logger_config.py                # Loguru setup, per-module log files
│   ├── error_handling.py               # Retry policies, resilient I/O, LLM wrapper
│   ├── prompt_templates.py             # Six LLM prompt categories
│   ├── data_processor.py               # Load, validate, quality check, profile
│   ├── dashboard_analyzer.py           # LLM analysis + rule-based fallbacks
│   ├── visualization_recommender.py    # Rule-based chart type selection
│   ├── tableau_workbook_generator.py   # Tableau XML generation (~1500 lines)
│   └── workflow.py                     # LangGraph pipeline orchestration
├── output/                             # Generated .twb and .twbx files land here
├── logs/                               # Per-module log files + intermediate JSON
└── requirements.txt
```

---

## What Each File Does

### `schemas.py` — The Data Contract

This is the backbone. Every other file imports from it. It defines:

- **Type aliases** — `ColumnSemanticType`, `ChartType`, `FileFormat`, `Severity` as `Literal` types so Pydantic rejects invalid values at construction time.
- **ColumnSchema / DatasetSchema** — Per-column metadata (name, raw dtype, semantic type, missing count, unique count, sample values) and the full dataset descriptor. All models use `ConfigDict(extra="allow")` so unexpected LLM output fields don't crash the pipeline.
- **Quality models** — `MissingValueReport`, `DuplicateReport`, `OutlierReport`, `TypeIssue`, and the top-level `QualityReport` with a composite 0–100 quality score.
- **Profile models** — `NumericProfile`, `CategoricalProfile`, `DatetimeProfile`, `TextProfile`, wrapped in `ColumnProfile` and collected in `ProfileReport`.
- **Visualization models** — `VisualizationSpec` (chart type, x/y columns, aggregation, rationale, confidence score) and `DashboardSpec` (list of visuals + layout).
- **Workbook models** — `WorksheetSpec` (Tableau shelf assignments: mark type, rows shelf, columns shelf, color/size fields) and `WorkbookSpec` (datasource name, worksheets, dashboard dimensions).
- **Workflow models** — `WorkflowConfig` (user preferences), `WorkflowState` (full pipeline state), `WorkflowProgress`, `WorkflowError`, `AnalysisResult`, `KPIRecommendation`.
- **Builder functions** — `build_dataset_schema()`, `build_quality_report()`, `build_profile_report()`, `build_dashboard_spec()` for quick construction from raw DataFrames. The dashboard spec builder has its own chart heuristics (datetime + numeric → line, one numeric → histogram, two numerics → scatter, categorical + numeric → bar/pie/treemap depending on cardinality).

### `logger_config.py` — Centralized Logging

Uses [Loguru](https://github.com/Delgan/loguru) instead of stdlib logging. The `get_logger(module_name)` function creates a bound logger with a per-module rotating file sink (daily rotation, 7-day retention) and a colored stderr sink. Every module calls this function once at import time. No duplicate sinks are created because it tracks registered module names in a set.

### `error_handling.py` — Resilience Layer

Three components:

- **`RetryPolicy` + `retry()`** — A frozen dataclass defining max attempts, exponential backoff with jitter, and which exception types to catch. The `retry()` function wraps any callable and handles the sleep/retry loop.
- **Resilient file readers** — `safe_read_csv_with_fallbacks()` tries UTF-8 then latin-1 encoding, with retries on each. `safe_read_excel_with_fallbacks()` tries the default engine then openpyxl.
- **`resilient_json_llm_call()`** — Wraps LLM provider calls with retry logic, strips markdown code fences from responses, parses JSON, and supports a two-tier fallback: cached previous response → empty dict. `LLMTransientError` is a custom exception that wraps provider errors (429, 5xx, timeout) so the retry policy catches them.

### `prompt_templates.py` — LLM Prompts

Six prompt-building functions, each returning `{"system": ..., "user": ...}`:

1. **`dataset_overview_prompt`** — Identify domain, grain, time coverage, key entities, column roles.
2. **`kpi_recommendation_prompt`** — Recommend primary KPIs (with Tableau formulas), secondary metrics, trend metrics, comparative metrics.
3. **`visualization_recommendation_prompt`** — Chart types with shelf mappings, color palettes, accessibility notes.
4. **`data_quality_assessment_prompt`** — Quality score, grade, blocking/non-blocking issues.
5. **`business_narrative_prompt`** — Dashboard title, executive summary, story arc, insight hypotheses.
6. **`dashboard_layout_prompt`** — Pixel-level zone placement for a 1200×900 canvas.

Currently only prompts 1 and 2 are wired into the production pipeline. The others are built and tested for future integration.

### `data_processor.py` — Data Ingestion

The `DataProcessor` class has four methods that form a mini-pipeline:

- **`load()`** — Reads CSV/Excel with encoding fallback. Strips whitespace from column names.
- **`validate()`** — Enforces minimums (≥1 row, ≤500 columns), runs semantic type inference on every column. The type detector (`_infer_semantic_type()`) uses an 80% threshold for coercion-based detection — a column needs 80% of its values to successfully parse as numeric or datetime before we reclassify it. Cardinality threshold of 50 separates categorical from text.
- **`assess_quality()`** — Missing values, duplicates, IQR-based outlier detection (1.5× multiplier), empty/constant columns, type issues (non-numeric strings in numeric columns, unparseable dates). Produces a composite score: starts at 100, deducts up to 30 for missing data, 20 for duplicates, 15 for empty columns, 9 for constant columns, 15 for type issues.
- **`profile()`** — Per-column statistical profiles. Numeric gets pandas `.describe()`, categorical gets value counts and top-k, datetime gets the date range, text gets string length stats.

### `dashboard_analyzer.py` — The AI Engine

The `DashboardAnalyzer` class has two public methods:

- **`analyze()`** — Builds a text profile of the DataFrame, sends it to the LLM via `dataset_overview_prompt`, and returns domain/grain/column roles. On failure, `_fallback_overview()` uses keyword-based column classification: geographic keywords → geographic, date keywords → temporal, high-cardinality ID-like columns → identifiers, numeric with ≤20 unique values → dimension, numeric with >20 → measure.
- **`recommend_kpis()`** — Sends the profile + analysis results to the LLM via `kpi_recommendation_prompt`. On failure, `_fallback_kpis()` generates SUM and AVG for each measure, COUNT of total records, trend metrics if temporal columns exist, and comparative metrics for each dimension × first measure. It even checks for money-related keywords in column names to set appropriate display formats.

The LLM client is lazily initialized — it only imports and creates the Azure/OpenAI client when the first LLM call is made. Retry logic uses `resilient_json_llm_call()` from the error handling module. Token usage and estimated cost are tracked across calls.

### `visualization_recommender.py` — Chart Selection

Entirely rule-based, entirely deterministic. Five rules fire in order:

1. **Time series** — Datetime + numeric → line charts (up to 3 numeric columns). If a low-cardinality categorical column exists (3–10 unique values), use it as group_by.
2. **Categorical breakdowns** — Score categorical columns by cardinality (skip single-value and >50). Bar charts for the top 3 categories. Pie charts only when cardinality ≤ 7.
3. **Distributions** — Histograms for up to 2 numeric columns. Box plot if a grouping column exists.
4. **Numeric relationships** — Scatter plots for up to 3 pairs of numeric columns (needs ≥30 rows).
5. **Table fallback** — If fewer than 2 visuals were generated, add a summary table.

Aggregation guessing: column names with "rate", "score", "avg" → AVG; "count", "qty" → SUM; values in [0,1] → AVG; default → SUM. The grouping column picker prefers columns with ~5 unique values (sweet spot for clean legends).

### `tableau_workbook_generator.py` — The XML Builder

At ~1500 lines, this is where abstract specs become real Tableau files. Three stages:

**Stage 1: `_plan_worksheets()`** — Converts each `VisualizationSpec` into a `WorksheetSpec` with concrete shelf assignments. Bar/line: x → Columns, y → Rows. Pie: measure → Rows + Size, dimension → Color. Histogram: numeric → Columns only (Tableau auto-bins). Scatter: both axes on their respective shelves. Table: group_by → Rows. Deduplicates worksheet names with a counter suffix.

The generator also has a date column fallback detector: scans column names for date keywords, tries parsing a sample of 200 values with `dayfirst=True`, and if ≥80% parse successfully, injects time-series line charts even if the upstream schema missed the dates.

**Stage 2: `_build_xml()`** — Assembles the full Tableau XML tree. The element ordering matches real Tableau-generated .twb files (reverse-engineered from manually created workbooks):
- `<document-format-change-manifest>` — Feature flags Tableau expects
- `<preferences>` — Shelf height defaults
- `<datasources>` — Federated datasource with named connection, relation with per-column type declarations, metadata records (remote-type codes: 5=real, 129=string, 133=date), column role definitions
- `<worksheets>` — Each with view/datasource-dependencies (column + column-instance elements), panes with mark type and encodings, rows/cols shelf expressions
- `<dashboards>` — Layout zones referencing worksheets (only if multiple worksheets)
- `<windows>` — Tells Tableau which sheet to display on open

The namespace registration (`ET.register_namespace("user", "...")`) is necessary because Tableau uses a custom XML namespace and without it, ElementTree serializes attributes as `ns0:ui-domain` which Tableau rejects.

Column instance references follow Tableau's convention: `[datasource].[agg:FieldName:typekey]` where typekey is `nk` (nominal/string), `ok` (ordinal/date), or `qk` (quantitative/numeric).

**Stage 3: Export** — `export_twb()` writes pretty-printed XML. `export_twbx()` creates a ZIP archive containing the .twb, the source CSV, and a .hyper extract (generated via pantab wrapping Tableau's Hyper API). After writing the .twbx, the extract element is removed from the in-memory XML tree so a subsequent `export_twb()` call doesn't reference a non-existent .hyper file.

**Validation** — Seven checks: root tag, version attribute, at least one datasource, worksheets with names, dashboard presence, all shelf-referenced columns exist in the dataset, dashboard zones reference valid worksheets.

**Calculated fields** — `add_calculated_field()` lets you add custom Tableau formulas (like Profit Ratio = `SUM([Profit]) / SUM([Sales])`) before export. If the XML was already built, it forces a rebuild.

### `workflow.py` — Pipeline Orchestration

The `DashboardGeneratorWorkflow` class uses LangGraph's `StateGraph` to chain the six node functions. A `GraphState` TypedDict defines the shape of the shared state — without this, LangGraph's merge logic would only keep keys from the last node.

Key implementation details:
- The DataFrame is serialized to JSON (`orient="split"`) in the state so it can travel through LangGraph's state merging. A helper `_rebuild_df()` deserializes it back in each node that needs it.
- `analyze_node` passes the DataFrame directly to `DashboardAnalyzer` instead of making it re-read the file from disk.
- `recommend_node` starts with `build_dashboard_spec()` as a baseline, then layers on charts from the analyzer's KPI recommendations (trend_metrics and comparative_metrics). This way the LLM's suggestions actually show up in the final workbook.
- `generate_node` syncs any worksheets the generator injected internally (date fallback charts) back into the dashboard_spec so the pipeline state reflects what's actually in the .twb file.
- `run_step_by_step()` uses LangGraph's `stream()` to fire an `on_progress` callback after each node — this is what drives Streamlit's progress bar.

### `app.py` — Streamlit Front-End

Four-page Streamlit application:

1. **Upload** — Drag-and-drop with 200 MB limit. Shows row count, column count, missing cells, duplicates. 20-row preview table and expandable column details.
2. **Configure** — Provider dropdown (Gemini / Azure), model name, API key (password-masked), endpoint URL. If left blank, defaults to rule-based heuristics.
3. **Process** — Instantiates the workflow, passes LLM credentials through WorkflowConfig's extra fields, runs `run_step_by_step()` with a callback that updates a progress bar. Pipeline logs are captured in a StringIO buffer and shown in an expander.
4. **Results** — Download buttons for .twb and .twbx. DashboardSpec as JSON. Error/warning list.

Logging is wired into both stdlib and Loguru so every module's output shows up in the Streamlit log panel.

---

## Installation

## How to Run the Project

1. Clone or download the project.

2. Open the project folder in terminal:

```bash
cd Capstone-AI-Tableau-Generator

# Install the required packages:
python -m pip install -r 0.requirements.txt
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Run the Streamlit app:
python -m streamlit run app.py
```

### Core Dependencies

| Package | What It's For |
|---------|--------------|
| `pandas` | DataFrame operations, CSV/Excel reading |
| `numpy` | Numeric computations in profiling and quality checks |
| `pydantic` | Data validation for every model in schemas.py |
| `langgraph` | Pipeline orchestration (state graph, node chaining) |
| `loguru` | Structured logging with per-module file rotation |
| `streamlit` | Web interface |
| `openai` | LLM API client (works with Azure OpenAI and Gemini via OpenAI-compatible endpoints) |

### Optional Dependencies

| Package | What It's For |
|---------|--------------|
| `pantab` | Generating .hyper extracts for .twbx files |
| `tableauhyperapi` | Required by pantab for Hyper file creation |
| `openpyxl` | Fallback Excel reader |

If pantab isn't installed, the pipeline still works — it just skips .twbx generation and produces only the .twb file.

---

## Usage

### Web Interface (Recommended)

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`). Upload your file, optionally configure LLM settings, run the pipeline, and download the results.

### Python API

```python
from scripts.workflow import DashboardGeneratorWorkflow
from scripts.schemas import WorkflowConfig

wf = DashboardGeneratorWorkflow()

# Basic usage (rule-based only)
result = wf.run("your_data.csv")
print(result["output_path"])  # output/your_data_dashboard.twb

# With LLM analysis
config = WorkflowConfig(use_ai_analysis=True)
config.llm_api_key = "your-api-key"
config.llm_provider = "gemini"       # or "azure"
config.llm_model = "gemini-3-flash-preview"
# config.llm_endpoint = "https://..."  # required for Azure

result = wf.run("your_data.csv", config=config)
```

### Individual Modules

Each module can be run standalone for testing:

```bash
# Test data processing
python -m scripts.data_processor

# Test the analyzer (rule-based)
python scripts/dashboard_analyzer.py your_data.csv

# Test visualization recommendations
python scripts/visualization_recommender.py

# Test Tableau XML generation
python scripts/tableau_workbook_generator.py your_data.csv
```

---

## Configuration Options

| Setting | Default | What It Controls |
|---------|---------|-----------------|
| `use_ai_analysis` | `True` | Whether to attempt LLM calls (falls back to rules if no API key) |
| `max_visualizations` | `8` | Maximum charts on the dashboard (1–20) |
| `quality_threshold` | `40.0` | Minimum quality score to proceed (0–100) |
| `output_format` | `"twb"` | Output format (`"twb"` or `"twbx"`) |
| `preferred_chart_types` | `[]` | Bias toward specific chart types |
| `business_goal` | `None` | Free-text business objective for LLM context |

---

## Design Decisions Worth Knowing

**Why LangGraph instead of plain function calls?**
LangGraph gives us automatic state merging (each node only returns changed keys), a visual graph representation (saved as PNG), and the `stream()` API for real-time progress callbacks. We could chain functions manually, but we'd lose state isolation and the Streamlit integration would be harder.

**Why generate Tableau XML directly?**
The Tableau REST API requires a running Tableau Server. The Hyper API only creates data extracts, not workbook structure. A .twb is just XML — by generating it directly, we produce a portable file that works on any machine with Tableau Desktop. No server needed.

**Why rule-based visualization recommendations instead of ML?**
Determinism. Same input always gives the same output. Every recommendation comes with a rationale string explaining why it was chosen. No training data or extra dependencies needed. Easy to extend when new chart types are added.

**Why Pydantic everywhere?**
Runtime validation catches bad data early. `extra="allow"` on all models means the LLM can return unexpected keys without crashing. Field validators (like confidence between 0 and 1, quality score non-negative) enforce constraints at construction time.

**Why Loguru instead of stdlib logging?**
Per-module rotating log files, colored terminal output, and structured context — all in about 30 lines of setup code. No manually configuring handlers and formatters.

**How was the Tableau XML structure figured out?**
We created dashboards manually in Tableau Desktop, saved them as .twb files, opened the XML in a text editor, and matched every required tag. Tableau doesn't publish an XSD schema, so the structure was reverse-engineered empirically. We also referenced the community-documented schema at [github.com/ranvithm/tableau.xml](https://github.com/ranvithm/tableau.xml).

---

## Edge Cases and How They're Handled

| Situation | What Happens |
|-----------|-------------|
| LLM returns invalid JSON | Strips markdown fences, retries up to 3× with backoff, falls back to cached response, then to rule-based heuristics |
| LLM rate limited (429) | Exponential backoff with jitter (0.8s base, 2× factor, ±0.35s jitter, 10s max) |
| CSV with non-UTF-8 encoding | Retries with latin-1 automatically |
| Date columns stored as DD/MM/YYYY strings | `_detect_date_columns_fallback()` scans for them and injects line charts |
| Numeric ID columns (Row ID, Postal Code) | Filtered out of measure lists to prevent nonsense charts |
| Duplicate worksheet names | Counter suffix: "Sales by Region", "Sales by Region (2)" |
| All columns are text | Table fallback generates a summary table so the dashboard isn't empty |
| pantab not installed | .twbx generation is skipped gracefully, .twb still produced |
| Dataset with 0 numeric columns | Only table and categorical charts are generated |
| Quality score below threshold | Pipeline marks the run as failed but still produces partial output |

---

## Output

The pipeline generates files in the `output/` directory:

- **`<dataset>_dashboard.twb`** — Plain Tableau workbook XML. Open directly in Tableau Desktop. Points to the original data file via a relative path.
- **`<dataset>_dashboard.twbx`** — Packaged workbook. Contains the .twb, the source CSV, and a .hyper extract. Self-contained and portable — you can share it without sending the data file separately.

Intermediate results are dumped to `logs/` as JSON for debugging:
- `output_results.json` — Schema, quality report, profile report
- `output_dashboard_analyzer.json` — Analysis and KPI recommendations
- `recommendations_output.json` — Visualization specs

---

## Supported Chart Types

| Chart Type | When It's Used | Tableau Mark |
|-----------|---------------|-------------|
| Line | Datetime + numeric (time series) | Line |
| Bar | Categorical + numeric (comparison) | Bar |
| Pie | Categorical (≤7 values) + numeric | Pie |
| Histogram | Single numeric (distribution) | Bar |
| Box | Numeric grouped by categorical | Circle |
| Scatter | Two numeric columns (correlation) | Circle |
| Heatmap | Two dimensions + measure | Square |
| Treemap | High-cardinality categorical + numeric | Square |
| Table | Fallback for sparse datasets | Text |

---

## Limitations

- The generated XML targets Tableau Desktop 2024.1+ (version 18.1). Backwards compatibility with older versions hasn't been tested.
- Geographic visualizations (maps) aren't generated yet — the column detection identifies geographic columns, but map chart generation isn't implemented.
- The .twbx Hyper extract requires pantab and tableauhyperapi, which need platform-specific binaries.
- Only two of the six LLM prompt templates are currently used in production. The visualization, quality, narrative, and layout prompts are built but not yet wired into the pipeline.

