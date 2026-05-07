# AI-Powered Tableau Dashboard Generator

Drop a CSV or Excel file in, get a ready-to-open Tableau workbook out. No manual chart building, no shelf dragging, no formula writing.

This project takes any tabular dataset, figures out what the data is about, picks the right KPIs and chart types, and generates valid `.twb` and `.twbx` files that open directly in Tableau Desktop. It works with or without an LLM — if you plug in an API key (Azure OpenAI or Google Gemini), the analysis gets smarter, but if you don't, it falls back to deterministic heuristics and still produces a solid dashboard.


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

#
## Installation

## How to Run the Project

1.  download the project.

2. Open the project folder in terminal in VSCode

```bash
cd Capstone-AI-Tableau-Generator
python -m venv venv
# venv\Scripts\activate 

# Install the required packages:
python -m pip install -r 0.requirements.txt


# Run the Streamlit app:
python -m streamlit run app.py
```



## Usage

### Web Interface (Recommended)

```bash
streamlit run app.py
```


## Output

The pipeline generates files in the `output/` directory:

- **`<dataset>_dashboard.twb`** — Plain Tableau workbook XML. Open directly in Tableau Desktop. Points to the original data file via a relative path.
- **`<dataset>_dashboard.twbx`** — Packaged workbook. Contains the .twb, the source CSV, and a .hyper extract. Self-contained and portable — you can share it without sending the data file separately.

Intermediate results are dumped to `logs/` as JSON for debugging:
- `output_results.json` — Schema, quality report, profile report
- `output_dashboard_analyzer.json` — Analysis and KPI recommendations
- `recommendations_output.json` — Visualization specs
