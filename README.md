# AI-Powered Tableau Dashboard Generator

This project takes any excel dataset, understands what the data is about, picks the right KPIs and chart types, and generates valid `.twb` and `.twbx` files that open directly in Tableau Desktop. It works with or without an LLM — if you give API key (Azure OpenAI or Google Gemini), the analysis using AI, but if you don't, it falls back to deterministic heuristics and still produces a solid dashboard.


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
|   └── test_dashboard_analyzer.py      # Unit Test file
|   └── test_data_processor.py          # Unit Test file
|   └── test_main_file_state.py          # Unit Test file
|   └── test_visualization_recommender.py  # Unit Test file
|   └── test_workflow_end_to_end.py      # Unit Test file
├── output/                             # Generated .twb and .twbx files land here
├── logs/                               # Per-module log files + intermediate JSON
├── datasets/                            # has the datasets
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

venv\Scripts\activate 

# Install the required packages:
pip install -r requirements.txt


# Run the Streamlit app(open app.py in vscode)
streamlit run app.py
```
