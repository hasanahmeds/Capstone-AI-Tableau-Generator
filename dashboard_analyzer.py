"""
dashboard_analyzer.py
=====================
AI Analysis Engine for the AI-Powered Tableau Dashboard Generator.

This code Contains the DashboardAnalyzer class which uses an LLM (Azure OpenAI /Gemini or
any compatible provider) to analyze datasets and generate structured
recommendations. Falls back to rule-based heuristics on API failure.

Usage:
    from dashboard_analyzer import DashboardAnalyzer

    # If you want to use with LLM
    analyzer = DashboardAnalyzer(api_key="...", endpoint="...", model="gpt-4o")
    analyzer.load_data("train.csv")
    overview = analyzer.analyze()
    kpis     = analyzer.recommend_kpis()

    # if you want to use without LLM (rule-based fallback)
    analyzer = DashboardAnalyzer()
    analyzer.load_data("train.csv")
    overview = analyzer.analyze()
    kpis     = analyzer.recommend_kpis()
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd

from prompt_templates import (
    dataset_overview_prompt,
    kpi_recommendation_prompt,
)
from error_handling import (
    RetryPolicy,
    resilient_json_llm_call,
    safe_read_csv_with_fallbacks,
    safe_read_excel_with_fallbacks,
    simple_profile_cache_key,
    LLMTransientError,
)

# added Logger for future use 
# ------------- 
logger = logging.getLogger("dashboard_analyzer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(name)s — %(message)s"))
    logger.addHandler(_h)


# DashboardAnalyzer Function
# ---------------------------

class DashboardAnalyzer:
    """Analyze a dataset and recommend KPIs for a Tableau dashboard.

    Two core public methods:
        analyze(): This returns a dict describing dataset domain, grain,
                           column roles, entities, and observations.
        recommend_kpis() : This returns a dict of primary KPIs, secondary metrics,
                           trend metrics, and comparative metrics.

    Both methods try the LLM first; on failure they fall back to
    deterministic rule-based heuristics so the pipeline never blocks.
    """

    # Keywords used by the rule-based fallback to classify columns testing with only one dataset might add in the future for other datasets if needed
    _GEO_KW  = {"country","state","city","region","zip","postal","latitude",
                 "longitude","lat","lng","lon","address","province","county"}
    _DATE_KW = {"date","time","datetime","timestamp","year","month","day",
                "week","quarter","period","created","updated"}
    _ID_KW   = {"id","key","code","number","no","num"}

    # Init function
    # ------------------------------------------------------------------

    def __init__(
        self,
        api_key: str  = None,
        endpoint: str = None,
        model: str    = "Ex: gpt-4o or gemini-3-flash-preview",
        provider: str = "azure/gemini",       # "azure open ai/gemini
        api_version: str = "2024-06-01",
        max_retries: int = 3,
        timeout: int     = 60,
    ):
        """
        Argument:
            api_key:     LLM API key.  if there is no api key then it's rule-based only.
            endpoint:    API endpoint URL (required for Azure OpenAI or Gemini).
            model:       Model / deployment name.
            provider:    "azure openai" or "gemini".
            api_version: Azure / Gemini API version string.
            max_retries: Retry count on LLM failure.
            timeout:     HTTP timeout in seconds.
        """
        self.api_key     = api_key
        self.endpoint    = endpoint
        self.model       = model
        self.provider    = provider
        self.api_version = api_version
        self.max_retries = max_retries
        self.timeout     = timeout

        self.df       = None          # loaded DataFrame
        self.overview = None          # cached analyze() result
        self.kpi_result = None        # cached recommend_kpis() result

        self._llm_client = None       # lazy-init OpenAI / Geminiclient
        self._profile_cache = {}      # hash: profile string

        self.total_tokens = 0         # cumulative token counter for prompt
        self.total_cost   = 0.0       # cumulative estimated cost (USD) approximate value

        logger.info(
            "DashboardAnalyzer initialized "
            f"(LLM={'enabled' if api_key else 'disabled — fallback only'})"
        )

    
    # Data loading to check the functions execution
    # ---------------------------------------------

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load a CSV or Excel file into self.df.

        Automatically parses columns whose names contain date-like keywords.
        Uses resilient readers with encoding fallbacks and retries from
        error_handling module.

        Arguments:
            filepath: Path to .csv or .xlsx file.
        Returns:
            The loaded DataFrame.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path.suffix.lower() == ".csv":
            self.df = safe_read_csv_with_fallbacks(path)
        elif path.suffix.lower() in (".xlsx", ".xls"):
            self.df = safe_read_excel_with_fallbacks(path)
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

        # auto-parse date columns
        for col in self.df.columns:
            if any(kw in col.lower() for kw in self._DATE_KW):
                try:
                    self.df[col] = pd.to_datetime(self.df[col],
                                                   errors="coerce",
                                                   dayfirst=True)
                except Exception:
                    pass

        logger.info(f"Loaded {path.name}: "
                     f"{self.df.shape[0]} rows × {self.df.shape[1]} cols")
        return self.df

    
    # Function : analyze()
    # -------------------

    def analyze(self) -> dict:
        """Analyze the loaded dataset and return a structured overview.

        Returns a dict with keys:
            dataset_domain, dataset_description, grain, time_coverage,
            key_entities, column_roles, notable_observations

        Falls back to rule-based classification if the LLM call fails
        or no API key was provided.
        """
        if self.df is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        logger.info("Running analyze()...")
        profile = self._build_profile()

        # --- try LLM ---
        if self.api_key:
            try:
                prompt = dataset_overview_prompt(profile)
                result = self._call_llm(prompt)
                self.overview = result
                logger.info(f"LLM analyze() done — domain: "
                             f"{result.get('dataset_domain')}")
                return self.overview
            except Exception as e:
                logger.warning(f"LLM analyze() failed, using fallback: {e}")

        # ---if LLM not mentioned or failed using the rule-based fallback ---
        self.overview = self._fallback_overview()
        return self.overview
    
    # Function METHOD : recommend_kpis()
    # -------------------------------------

    def recommend_kpis(self) -> dict:
        """Recommend KPIs and metrics for a Tableau dashboard.

        Returns a dict with keys:
            primary_kpis, secondary_metrics, trend_metrics,
            comparative_metrics

        Automatically calls analyze() first if it hasn't been run yet.
        Falls back to rule-based generation on LLM failure.
        """
        if self.df is None:
            raise RuntimeError("No data loaded. Call load_data() first.")

        if self.overview is None:
            self.analyze()

        logger.info("Running recommend_kpis()...")
        profile = self._build_profile()

        # --- try LLM ---
        if self.api_key:
            try:
                prompt = kpi_recommendation_prompt(
                    profile_summary=profile,
                    dataset_domain=self.overview.get("dataset_domain", ""),
                    column_roles=self.overview.get("column_roles", {}),
                )
                result = self._call_llm(prompt)
                self.kpi_result = result
                logger.info(
                    f"LLM recommend_kpis() done — "
                    f"{len(result.get('primary_kpis',[]))} primary, "
                    f"{len(result.get('secondary_metrics',[]))} secondary"
                )
                return self.kpi_result
            except Exception as e:
                logger.warning(f"LLM recommend_kpis() failed, "
                                f"using fallback: {e}")

        # --- if LLM not mentioned or failed using the rule-based fallback ---
        self.kpi_result = self._fallback_kpis()
        return self.kpi_result

    
    # Function : LLM call with retry
    # ----------------------------

    def _get_llm_client(self):
        """Lazy-initialize the OpenAI / Azure client."""
        if self._llm_client is not None:
            return self._llm_client

        if self.provider == "azure":
            from openai import AzureOpenAI
            self._llm_client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                timeout=self.timeout,
            )
        else:
            from openai import OpenAI
            kw = {"api_key": self.api_key, "timeout": self.timeout}
            if self.endpoint:
                kw["base_url"] = self.endpoint
            self._llm_client = OpenAI(**kw)

        return self._llm_client

    def _call_llm(self, prompt: dict) -> dict:
        """Call the LLM using resilient retry + JSON parsing from error_handling.

        Arguments:
            prompt: dict with 'system' and 'user' keys.
        Returns:
            Parsed JSON dict from the LLM response.
        """
        client = self._get_llm_client()

        # Build a cache key from the prompt content for fallback lookup
        cache_key = simple_profile_cache_key(
            prompt.get("system", "") + prompt.get("user", "")
        )

        def _provider_call():
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user",   "content": prompt["user"]},
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                # Wrap transient provider errors so the retry policy catches them
                err_str = str(exc).lower()
                if any(kw in err_str for kw in ("timeout", "rate", "429", "500", "502", "503", "504")):
                    raise LLMTransientError(str(exc)) from exc
                raise
            elapsed = time.time() - t0

            # track tokens
            if resp.usage:
                self.total_tokens += resp.usage.total_tokens
                self.total_cost  += round(
                    resp.usage.prompt_tokens / 1000 * 0.005
                    + resp.usage.completion_tokens / 1000 * 0.015, 6
                )

            logger.info(f"LLM responded in {elapsed:.1f}s")
            return resp

        policy = RetryPolicy(
            max_attempts=self.max_retries,
            base_delay_s=1.0,
            max_delay_s=12.0,
            backoff=2.0,
            jitter_s=0.5,
            retry_on=(TimeoutError, ConnectionError, OSError, LLMTransientError),
        )

        result = resilient_json_llm_call(
            _provider_call,
            policy=policy,
            op_name="dashboard_llm_call",
            cache_get=lambda: self._profile_cache.get(f"llm_{cache_key}"),
            cache_set=lambda d: self._profile_cache.__setitem__(f"llm_{cache_key}", d),
            fallback_empty_json=False,
        )
        return result

    # Function — Profile builder
    # --------------------------

    def _build_profile(self) -> str:
        """Build a concise text profile of self.df (cached)."""
        key = simple_profile_cache_key(
            f"{self.df.shape}{list(self.df.columns)}"
        )
        if key in self._profile_cache:
            return self._profile_cache[key]

        df = self.df
        lines = [f"SHAPE: {df.shape[0]} rows × {df.shape[1]} columns\n"]

        # --- column details ---
        lines.append("COLUMNS:")
        for col in df.columns:
            nu = df[col].nunique()
            nm = int(df[col].isna().sum())
            pct = round(100 * nm / len(df), 1)
            samp = ", ".join(str(v) for v in df[col].dropna().head(3).tolist())
            lines.append(f"  {col}  |  dtype={df[col].dtype}  |  "
                          f"unique={nu}  |  missing={nm} ({pct}%)  |  "
                          f"sample=[{samp}]")

        # --- numeric summary ---
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if num_cols:
            lines.append("\nNUMERIC SUMMARY:")
            lines.append(df[num_cols].describe().round(2).to_string())

        # --- top values for low-cardinality categoricals ---
        cat_cols = [c for c in df.select_dtypes(
                        include=["object","category","string"]).columns
                    if df[c].nunique() <= 50]
        if cat_cols:
            lines.append("\nTOP VALUES (categorical):")
            for col in cat_cols[:10]:
                top = df[col].value_counts().head(3)
                vals = ", ".join(f"{v}({n})" for v, n in top.items())
                lines.append(f"  {col}: {vals}")

        # --- date ranges ---
        dt_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()
        if dt_cols:
            lines.append("\nDATE RANGES:")
            for col in dt_cols:
                lines.append(f"  {col}: {df[col].min()} → {df[col].max()}")

        lines.append(f"\nTOTAL MISSING: {int(df.isna().sum().sum())}")
        lines.append(f"DUPLICATE ROWS: {int(df.duplicated().sum())}")

        text = "\n".join(lines)
        self._profile_cache[key] = text
        return text

    # PRIVATE — Column classifier (rule-based)
    # -----------------------------------------------------

    def _classify_columns(self) -> dict:
        """Classify columns into roles using heuristic rules.

        Returns:
            dict with keys: dimensions, measures, temporal,
                            identifiers, geographic
        """
        df = self.df
        roles = {
            "dimensions": [], "measures": [], "temporal": [],
            "identifiers": [], "geographic": [],
        }

        for col in df.columns:
            cl = col.lower().replace(" ", "_")

            # identifiers (high-cardinality ID-like columns)
            if any(kw in cl for kw in self._ID_KW):
                if df[col].nunique() / max(len(df), 1) > 0.5:
                    roles["identifiers"].append(col)
                    continue

            # temporal
            if (pd.api.types.is_datetime64_any_dtype(df[col])
                    or any(kw in cl for kw in self._DATE_KW)):
                roles["temporal"].append(col)
                continue

            # geographic
            if any(kw in cl for kw in self._GEO_KW):
                roles["geographic"].append(col)
                roles["dimensions"].append(col)
                continue

            # numeric → measure or dimension
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() <= 20:
                    roles["dimensions"].append(col)
                else:
                    roles["measures"].append(col)
            else:
                roles["dimensions"].append(col)

        return roles

    # Function Fallback: analyze()
    # -----------------------------

    def _fallback_overview(self) -> dict:
        """Rule-based dataset overview (no LLM needed)."""
        df = self.df
        roles = self._classify_columns()

        # Check time coverage
        time_cov = {
            "has_time_dimension": False,
            "time_column": None,
            "estimated_range": None,
        }
        if roles["temporal"]:
            tc = roles["temporal"][0]
            time_cov["has_time_dimension"] = True
            time_cov["time_column"] = tc
            try:
                parsed = pd.to_datetime(df[tc], errors="coerce")
                time_cov["estimated_range"] = (
                    f"{parsed.min().year}-{parsed.max().year}"
                )
            except Exception:
                pass

        # Fetch key entities from ID columns
        #-----------------------------------
        entities = []
        for id_col in roles["identifiers"]:
            base = (id_col.lower()
                    .replace("_id", "").replace(" id", "").strip())
            name_col = next(
                (c for c in df.columns
                 if "name" in c.lower() and base in c.lower()),
                None,
            )
            entities.append({
                "entity_name": base.title(),
                "id_column": id_col,
                "name_column": name_col,
                "estimated_cardinality": f"~{df[id_col].nunique()} unique",
            })

        return {
            "dataset_domain": "Unknown (rule-based fallback)",
            "dataset_description":
                f"Dataset with {len(df)} rows and {len(df.columns)} columns.",
            "grain":
                f"One row per record ({len(df)} total rows)",
            "time_coverage": time_cov,
            "key_entities": entities,
            "column_roles": roles,
            "notable_observations": [
                f"Shape: {df.shape[0]} rows × {df.shape[1]} columns",
                f"Measures: {roles['measures']}",
                f"Temporal: {roles['temporal']}",
            ],
        }

    # Fallback Method: recommend_kpis()
    # ---------------------------------

    def _fallback_kpis(self) -> dict:
        """Rule-based KPI generation (no LLM needed)."""
        df = self.df
        roles = self.overview.get("column_roles", self._classify_columns())

        primary   = []
        secondary = []
        trends    = []
        comparative = []

        # --- primary + secondary from each measure column ---
        for m in roles.get("measures", [])[:5]:
            is_money = any(kw in m.lower()
                           for kw in ("sale","revenue","price","profit","cost"))
            primary.append({
                "name": f"Total {m}",
                "formula": f"SUM([{m}])",
                "source_columns": [m],
                "format": "$#,##0" if is_money else "#,##0",
                "business_rationale": f"Aggregate view of {m}.",
            })
            secondary.append({
                "name": f"Average {m}",
                "formula": f"AVG([{m}])",
                "source_columns": [m],
                "format": "#,##0.00",
                "business_rationale": f"Per-record average of {m}.",
            })

        # record count
        first_col = df.columns[0]
        primary.append({
            "name": "Total Records",
            "formula": f"COUNT([{first_col}])",
            "source_columns": [first_col],
            "format": "#,##0",
            "business_rationale": "Overall transaction volume.",
        })

        # --- trend metrics ---
        if roles.get("temporal") and roles.get("measures"):
            for m in roles["measures"][:2]:
                trends.append({
                    "name": f"Monthly {m} Trend",
                    "formula": f"SUM([{m}])",
                    "time_grain": "month",
                    "source_columns": [roles["temporal"][0], m],
                })

        # --- comparative metrics ---
        dims = [d for d in roles.get("dimensions", [])
                if d not in roles.get("geographic", [])][:3]
        if roles.get("measures"):
            main = roles["measures"][0]
            for d in dims:
                comparative.append({
                    "name": f"{main} by {d}",
                    "measure": f"SUM([{main}])",
                    "compare_by": d,
                    "source_columns": [d, main],
                })

        return {
            "primary_kpis": primary,
            "secondary_metrics": secondary,
            "trend_metrics": trends,
            "comparative_metrics": comparative,
        }

# To check the ouput use the following prompt in Terminal :  python dashboard_analyzer.py train.csv

if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "train.csv"

    analyzer = DashboardAnalyzer()           
    analyzer.load_data(path)

    # Run the analysis and store the results
    analysis_results = analyzer.analyze()
    recommended_kpis = analyzer.recommend_kpis()

    # Consolidate data into one dictionary 
    output_data = {
        "analysis": analysis_results,
        "recommended_kpis": recommended_kpis
    }

    #Add the outputs to the JSON file
    output_filename = "output_dashboard_analyzer.json"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Successfully saved analysis to {output_filename}")
    except Exception as e:
        print(f"Error saving to file: {e}")

    # Adding print to show the end of the Code exectution
    print("RESULTS SAVED TO FILE")
    print("=" * 60)