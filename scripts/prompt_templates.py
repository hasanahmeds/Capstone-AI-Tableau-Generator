"""
LLM Prompt Templates for project AI-Powered Tableau Dashboard Generator.


This code contains all the prompt templated we thing will be used by Dashboard Analyzer to interact with LLM. 
we may add more if we think we get better input in future.

Below are the template Categories:
    1. Dataset Overview Analysis
    2. KPI & Metrics Recommendation
    3. Visualization Recommendation
    4. Data Quality Assessment
    5. Business Insights & Narrative
    6. Dashboard Layout Recommendation
"""



# General instruction for all prompts

SYSTEM_PROMPT = (
    '''You are an expert data analyst and Tableau dashboard architect.
    You analyze datasets and provide actionable, structured recommendations
    for building professional Tableau dashboards. Always respond with valid
    JSON matching the schema described in the user prompt. Please do not include
    markdown code fences or any text outside the JSON object.'''
)

# Template Category 1 : Dataset Overview Analysis
#------------------------------------------------
def dataset_overview_prompt(profile_summary: str) -> dict:
    """Analyze overall structure, domain, grain, and column roles.

    Arguments:
        profile_summary: Text summary with column names, dtypes, stats,
                         sample values, missing counts, etc.
    Returns:
        dict with 'system' and 'user' keys ready for the LLM call.
    """
    user = f"""Analyze the following dataset profile and provide a structured overview.

### DATASET PROFILE ###
{profile_summary}

### INSTRUCTIONS ###
Return a JSON object with EXACTLY this schema:

{{
    "dataset_domain": "<detected business domain, e.g. 'Retail Sales'>",
    "dataset_description": "<1-2 sentence summary of what this dataset represents>",
    "grain": "<granularity of each row, e.g. 'one row per order line item'>",
    "time_coverage": {{
        "has_time_dimension": true/false,
        "time_column": "<column name or null>",
        "estimated_range": "<e.g. '2015-2018' or 'unknown'>"
    }},
    "key_entities": [
        {{
            "entity_name": "<e.g. 'Customer', 'Product'>",
            "id_column": "<column name>",
            "name_column": "<column name or null>",
            "estimated_cardinality": "<e.g. '793 unique'>"
        }}
    ],
    "column_roles": {{
        "dimensions": ["<columns used for grouping/slicing>"],
        "measures": ["<numeric columns suitable for aggregation>"],
        "temporal": ["<date/time columns>"],
        "identifiers": ["<ID columns not useful for viz>"],
        "geographic": ["<location-related columns>"]
    }},
    "notable_observations": [
        "<any interesting pattern, skew, or issue spotted>"
    ]
}}
"""
    return {"system": SYSTEM_PROMPT, "user": user}

 
# Template Category 2 : KPI & Metrics Recommendation
# -----------------------------------------
def kpi_recommendation_prompt(
    profile_summary: str,
    dataset_domain: str,
    column_roles: dict,
) -> dict:
    """Recommend KPIs and metrics for a Tableau dashboard.

    Args:
        profile_summary: Text profile of the dataset.
        dataset_domain:  Detected domain (e.g. 'Retail Sales').
        column_roles:    Dict mapping role names to lists of column names.
    Returns:
        Prompt dict.
    """
    user = f"""Given the dataset profile and context below, recommend the most
impactful KPIs and metrics for a Tableau dashboard.

### DATASET PROFILE ###
{profile_summary}

### CONTEXT ###
Domain      : {dataset_domain}
Dimensions  : {column_roles.get("dimensions", [])}
Measures    : {column_roles.get("measures", [])}
Temporal    : {column_roles.get("temporal", [])}
Geographic  : {column_roles.get("geographic", [])}

### INSTRUCTIONS ###
Return a JSON object with EXACTLY this schema:

{{
    "primary_kpis": [
        {{
            "name": "<human-readable KPI name, e.g. 'Total Revenue'>",
            "formula": "<Tableau calculation, e.g. 'SUM(Sales)'>",
            "source_columns": ["<columns involved>"],
            "format": "<display format, e.g. '$#,##0', '0.0%'>",
            "business_rationale": "<why this KPI matters>"
        }}
    ],
    "secondary_metrics": [
        {{
            "name": "<metric name>",
            "formula": "<calculation>",
            "source_columns": ["<columns>"],
            "format": "<display format>",
            "business_rationale": "<why useful>"
        }}
    ],
    "trend_metrics": [
        {{
            "name": "<e.g. 'Monthly Sales Trend'>",
            "formula": "<calculation>",
            "time_grain": "<day/week/month/quarter/year>",
            "source_columns": ["<columns>"]
        }}
    ],
    "comparative_metrics": [
        {{
            "name": "<e.g. 'Sales by Region'>",
            "measure": "<aggregated measure>",
            "compare_by": "<dimension column>",
            "source_columns": ["<columns>"]
        }}
    ]
}}

Guidelines:
- It Recommend 3-5 primary KPIs a C-level executive would care about.
- It Recommend 3-6 secondary metrics for deeper drill-down.
- At least 1 trend metric if temporal columns exist.
- At least 2 comparative metrics across key dimensions.
- Usea only columns that actually exist in the dataset.
"""
    return {"system": SYSTEM_PROMPT, "user": user}



# Template Category 3 : Visualization Recommendation
# ------------------------------------------
def visualization_recommendation_prompt(
    kpis: list,
    metrics: list,
    column_roles: dict,
    dataset_domain: str,
) -> dict:
    """Recommend chart types and Tableau shelf mappings for each KPI/metric.

    Args:
        kpis:           List of primary KPI dicts.
        metrics:        List of secondary metric dicts.
        column_roles:   Column role mapping.
        dataset_domain: Detected domain.
    Returns:
        Prompt dict.
    """
    user = f"""Recommend Tableau visualizations for the KPIs and metrics below.

### DOMAIN ###
{dataset_domain}

### KPIs ###
{kpis}

### SECONDARY METRICS ###
{metrics}

### AVAILABLE COLUMNS ###
Dimensions : {column_roles.get("dimensions", [])}
Measures   : {column_roles.get("measures", [])}
Temporal   : {column_roles.get("temporal", [])}
Geographic : {column_roles.get("geographic", [])}

### INSTRUCTIONS ###
Return a JSON object with EXACTLY this schema:

{{
    "visualizations": [
        {{
            "title": "<worksheet title>",
            "chart_type": "<one of: bar, stacked_bar, line, area, scatter, pie,
                           donut, treemap, heatmap, map, kpi_card, gauge,
                           histogram, combo, waterfall, bubble>",
            "columns_used": {{
                "rows": ["<fields on Rows shelf>"],
                "columns": ["<fields on Columns shelf>"],
                "color": "<field for color encoding or null>",
                "size": "<field for size encoding or null>",
                "tooltip": ["<extra tooltip fields>"]
            }},
            "aggregation": "<SUM / AVG / COUNT / COUNTD>",
            "sort_order": "<ascending / descending / none>",
            "rationale": "<why this chart type fits>"
        }}
    ],
    "color_palette": "<recommended Tableau palette, e.g. 'Color Blind'>",
    "accessibility_notes": "<color-blind friendliness guidance>"
}}

Guidelines:
- 6-10 visualizations covering all primary KPIs.
- At least one time-series chart if temporal columns exist.
- At least one geographic viz if geographic columns exist.
- Include KPI cards for headline numbers.
- Prefer bar/line over pie for comparing categories.
"""
    return {"system": SYSTEM_PROMPT, "user": user}



# Template Category 4 : Data Quality Assessment
# ------------------------------------

def data_quality_assessment_prompt(quality_report: str) -> dict:
    """Assess data quality issues and suggest remediation.

    Args:
        quality_report: Text report with missing-value counts,
                        duplicate counts, outlier flags, etc.
    Returns:
        Prompt dict.
    """
    user = f"""Assess the data quality of this dataset and provide actionable
recommendations for cleaning before dashboard creation.

### QUALITY REPORT ###
{quality_report}

### INSTRUCTIONS ###
Return a JSON object with EXACTLY this schema:

{{
    "overall_quality_score": <float 0-100>,
    "quality_grade": "<A / B / C / D / F>",
    "is_dashboard_ready": true/false,
    "blocking_issues": [
        {{
            "column": "<column name>",
            "issue": "<description>",
            "severity": "<critical / warning / info>",
            "recommended_action": "<what to do>"
        }}
    ],
    "non_blocking_issues": [
        {{
            "column": "<column name>",
            "issue": "<description>",
            "severity": "<warning / info>",
            "recommended_action": "<what to do>"
        }}
    ],
    "summary": "<1-2 sentence  quality summary>"
}}
"""
    return {"system": SYSTEM_PROMPT, "user": user}

# Template Category 5 : Business Insights & Narrative
# -------------------------------------------

def business_narrative_prompt(
    profile_summary: str,
    dataset_domain: str,
    kpis: list,
) -> dict:
    """Generate dashboard title, audience, story arc, and insight hypotheses.

    Args:
        profile_summary: Text profile of the dataset.
        dataset_domain:  Detected domain.
        kpis:            List of recommended KPI dicts.
    Returns:
        Prompt dict.
    """
    user = f"""Generate a business narrative and insight hypotheses for a Tableau
dashboard based on the dataset described below.

### DATASET PROFILE ###
{profile_summary}

### DOMAIN ###
{dataset_domain}

### RECOMMENDED KPIs ###
{kpis}

### INSTRUCTIONS ###
Return a JSON object with EXACTLY this schema:

{{
    "dashboard_title": "<suggested title>",
    "executive_summary": "<2-3 sentence subtitle>",
    "target_audience": "<who should use this dashboard>",
    "key_questions": [
        "<business question the dashboard should answer>"
    ],
    "insight_hypotheses": [
        {{
            "hypothesis": "<e.g. 'West region drives highest revenue'>",
            "test_with": "<which KPI / visualization to validate>"
        }}
    ],
    "recommended_filters": [
        {{
            "column": "<column name>",
            "filter_type": "<dropdown / slider / date_range>",
            "rationale": "<why this filter is useful>"
        }}
    ],
    "story_arc": [
        "<step 1: high-level overview>",
        "<step 2: dive deeper into trend>",
        "<step 3: compare segments>",
        "<step 4: identify outliers>"
    ]
}}
"""
    return {"system": SYSTEM_PROMPT, "user": user}


# Template Category 6 : Dashboard Layout Recommendation
# --------------------------------------------
def dashboard_layout_prompt(visualizations: list, dashboard_title: str) -> dict:
    """Recommend pixel-level spatial layout for the Tableau dashboard.

    Args:
        visualizations:  List of visualization spec dicts.
        dashboard_title: Title of the dashboard.
    Returns:
        Prompt dict.
    """
    viz_names = [v.get("title", "Untitled") for v in visualizations]

    user = f"""Recommend a Tableau dashboard layout for the following worksheets.

### DASHBOARD TITLE ###
{dashboard_title}

### WORKSHEETS ###
{viz_names}

### INSTRUCTIONS ###
Return a JSON object with EXACTLY this schema:

{{
    "layout": {{
        "width": 1200,
        "height": 900,
        "zones": [
            {{
                "worksheet": "<worksheet title>",
                "position": {{
                    "x": <pixels from left>,
                    "y": <pixels from top>,
                    "width": <int>,
                    "height": <int>
                }},
                "zone_type": "<kpi_banner / main_chart / sidebar / filter_bar>"
            }}
        ]
    }},
    "design_notes": {{
        "kpi_placement": "<where KPI cards go>",
        "filter_placement": "<where filters go>",
        "emphasis": "<which chart should dominate visually>"
    }}
}}

Guidelines:
- KPI cards in a horizontal banner at the top.
- Main time-series chart as the largest element.
- Categorical breakdowns in the middle row.
- Filters on the left sidebar or top-right.
"""
    return {"system": SYSTEM_PROMPT, "user": user}
