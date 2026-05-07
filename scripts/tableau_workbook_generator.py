"""
Tableau Workbook Generator
--------------------------
Converts visualization recommendations from the pipeline into actual
Tableau workbook files (.twb and .twbx).

A .twb file is just XML that describes data sources, worksheets,
dashboards, and calculated fields. A .twbx is a ZIP archive that
bundles the .twb together with the raw data file (CSV/Excel) so the
workbook is self-contained and portable.

This module reads from the Pydantic models defined in schemas.py
(DashboardSpec, VisualizationSpec, DatasetSchema) and produces
files that Tableau Desktop can open directly.

Usage:
    from scripts.tableau_workbook_generator import TableauWorkbookGenerator

    generator = TableauWorkbookGenerator(
        dashboard_spec=dashboard,
        dataset_schema=ds,
        dataframe=df,
    )
    # write a standalone .twb
    generator.export_twb("output/my_dashboard.twb")

    # or bundle data into a .twbx
    generator.export_twbx("output/my_dashboard.twbx", data_file="train.csv")

    # quick validation
    is_ok, issues = generator.validate()

Design notes:
    - Uses xml.etree.ElementTree (stdlib) instead of lxml because
      lxml requires a C library install and the Tableau XML schema
      doesn't need the extra features lxml provides.
    - The generated XML targets Tableau 2024.1 format (version 18.1).
      Older Tableau versions may still open it with a compatibility
      just checked in the version mentioned below.

Few Default tags are added like build number etc by creating a manual tableau dashboard
and inspecting the XML as Tableau doesn't open files that don't have these tags.
"""

import os
import shutil
import tempfile
import zipfile
import xml.etree.ElementTree as ET

# Register the Tableau "user" namespace globally so that ElementTree
# serializes attributes like user:ui-domain correctly instead of
# generating ns0:ui-domain (which Tableau doesn't recognize).
# Register the Tableau "user" namespace globally so that ElementTree
# serializes attributes like user:ui-domain correctly instead of
# generating ns0:ui-domain (which Tableau doesn't recognize).
ET.register_namespace("user", "http://www.tableausoftware.com/xml/user")
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pandas as pd

from scripts.schemas import (
    DatasetSchema,
    ColumnSchema,
    DashboardSpec,
    VisualizationSpec,
    WorksheetSpec,
    WorkbookSpec,
)

# Logger setup — logs to logs/tableau_generator.log via loguru


from scripts.logger_config import get_logger

logger = get_logger("tableau_generator")



#Tableau XML version tags and defaults

# Tableau workbook XML format version.  18.1 corresponds to
# Tableau Desktop 2026.1.0  We picked this because it's lastest
# enough to support all the chart types we generate but old enough
# that most people running can still open the file.
# we didn't check backwards compatibility with older versions.
TABLEAU_VERSION = "18.1"
TABLEAU_BUILD = "2026.1.0 (20261.26.0226.1626)"

# The default dashboard canvas size in pixels.  1200×900 is
# Tableau's "automatic" desktop size and works well on most
# monitors without scrolling.
DEFAULT_DASHBOARD_WIDTH = 1200
DEFAULT_DASHBOARD_HEIGHT = 900

# Maps our internal chart type names (from VisualizationSpec.chart_type)
# to the mark type string that Tableau XML expects inside <mark> elements.
# Not every Tableau mark has a 1:1 mapping to our types.
CHART_TYPE_TO_MARK = {
    "bar":       "Bar",
    "line":      "Line",
    "scatter":   "Circle",
    "histogram": "Bar",
    "box":       "Circle",
    "heatmap":   "Square",
    "pie":       "Pie",
    "treemap":   "Square",
    "table":     "Text",
}

# Aggregation strings used in Tableau calculated fields.  Our schema
# uses lowercase names; Tableau XML needs uppercase.
AGG_MAP = {
    "sum":    "SUM",
    "avg":    "AVG",
    "min":    "MIN",
    "max":    "MAX",
    "count":  "COUNT",
    "median": "MEDIAN",
}

# Semantic type → Tableau datatype + role mapping.  Tableau distinguishes
# between "dimension" and "measure" roles, and between "nominal",
# "ordinal", "quantitative" data types.  These defaults cover the
# common cases; edge cases (like a numeric zip code that should be
# treated as a dimension) are handled in _resolve_column_role().
SEMANTIC_TO_TABLEAU = {
    "numeric":     {"datatype": "real",     "role": "measure",   "type": "quantitative"},
    "categorical": {"datatype": "string",   "role": "dimension", "type": "nominal"},
    "datetime":    {"datatype": "date",     "role": "dimension", "type": "ordinal"},
    "boolean":     {"datatype": "boolean",  "role": "dimension", "type": "nominal"},
    "text":        {"datatype": "string",   "role": "dimension", "type": "nominal"},
    "unknown":     {"datatype": "string",   "role": "dimension", "type": "nominal"},
}



# Main Workbook generator class

class TableauWorkbookGenerator:
    """Converts a DashboardSpec + DatasetSchema into a Tableau workbook file.

    The generator works in three stages:
      1. _plan_worksheets()  — translates each VisualizationSpec into a
         WorksheetSpec with concrete shelf assignments.
      2. _build_xml()        — assembles the full Tableau XML tree.
      3. export_twb/twbx()   — writes the XML to disk (plain or zipped).

    Each stage can be called independently for debugging, but the
    export methods handle the full module automatically.
    """

    def __init__(
        self,
        dashboard_spec: DashboardSpec,
        dataset_schema: DatasetSchema,
        dataframe: pd.DataFrame,
    ):
        """
        Args:
            dashboard_spec:  The recommended visualizations and layout.
            dataset_schema:  Column metadata (types, roles, cardinality).
            dataframe:       The actual data — needed for column names
                             and to bundle into .twbx files.
        """
        self.dashboard_spec = dashboard_spec
        self.dataset_schema = dataset_schema
        self.df = dataframe

        # These get populated when we build the workbook
        self._workbook_spec: Optional[WorkbookSpec] = None
        self._xml_tree: Optional[ET.ElementTree] = None

        # Calculated fields that get added during XML generation.
        # add_calculated_field() appends to this list.
        self._extra_calc_fields: List[Dict[str, str]] = []

        # Index column schemas by name for lookup
        self._col_index: Dict[str, ColumnSchema] = {
            col.name: col for col in dataset_schema.columns
        }

        # Derive a clean, short name from the dataset for captions.
        # The schema's .name often holds a full file path like
        # "/mnt/user-data/uploads/train.csv" — we just want "train".
        raw_name = dataset_schema.source_filename or dataset_schema.name or "Dataset"
        self._friendly_name = Path(raw_name).stem.replace("_", " ").title()

        # The upstream schema detection can miss date columns when dates are
        # stored as strings in DD/MM/YYYY or other ambiguous formats.
        # We scan again here so time-series charts still get injected even if
        # the recommender didn't know about those date columns.
        self._detected_date_cols = self._detect_date_columns_fallback()

        logger.info(
            f"TableauWorkbookGenerator ready — "
            f"{len(dashboard_spec.visuals)} visuals, "
            f"{dataframe.shape[0]} rows × {dataframe.shape[1]} cols"
            + (f", detected date cols: {self._detected_date_cols}"
               if self._detected_date_cols else "")
        )


    # Stage 1: Plan worksheets from VisualizationSpec objects

    def _plan_worksheets(self) -> WorkbookSpec:
        """Convert each VisualizationSpec into a WorksheetSpec.

        This is where we decide what goes on the Rows shelf, what goes
        on the Columns shelf, and what mark type to use.  The mapping
        rules are straightforward:

          - bar:       x → Columns (dimension), y → Rows (measure)
          - line:      x → Columns (date), y → Rows (measure)
          - scatter:   x → Columns (measure), y → Rows (measure)
          - histogram: x → Columns (binned), count → Rows
          - pie:       x → color, y → angle (size)
          - heatmap:   x → Columns, y → Rows, measure → color
          - box:       x → Columns, y → Rows
          - table:     group_by columns → Rows

        We also sanitize worksheet names here because Tableau doesn't
        allow certain characters (slashes, brackets) in sheet names.
        """
        worksheets = []
        seen_names = set()

        for viz in self.dashboard_spec.visuals:
            # make sure each worksheet has a unique name
            base_name = self._sanitize_sheet_name(viz.title)
            ws_name = base_name
            counter = 2
            while ws_name in seen_names:
                ws_name = f"{base_name} ({counter})"
                counter += 1
            seen_names.add(ws_name)

            mark = CHART_TYPE_TO_MARK.get(viz.chart_type, "Bar")
            agg = AGG_MAP.get(viz.aggregation, "SUM") if viz.aggregation else None

            rows = []
            cols = []
            color_field = None
            size_field = None
            tooltip_fields = []

            # ---- shelf assignment by chart type ----
            if viz.chart_type == "pie":
                # Tableau pie needs: measure on Rows (so view has data),
                # dimension on Color (one slice per category),
                # measure on Size (slice angle).
                if viz.y and self._column_exists(viz.y):
                    rows.append(viz.y)
                    size_field = viz.y
                if viz.x and self._column_exists(viz.x):
                    color_field = viz.x

            elif viz.chart_type == "treemap":
                # Treemap: dimension on detail, measure on size + color
                if viz.x and self._column_exists(viz.x):
                    color_field = viz.x
                if viz.y and self._column_exists(viz.y):
                    rows.append(viz.y)
                    size_field = viz.y

            elif viz.chart_type == "heatmap":
                # Heatmap: both axes are dimensions, measure goes on color
                if viz.x and self._column_exists(viz.x):
                    cols.append(viz.x)
                if viz.y and self._column_exists(viz.y):
                    rows.append(viz.y)
                    color_field = viz.y

            elif viz.chart_type == "histogram":
                # Histogram: single numeric column on Columns shelf only.
                # Tableau will auto-bin and count. Do NOT add "Number of Records"
                # to rows — that field doesn't exist in the datasource and
                # causes "field does not exist" errors.
                if viz.x and self._column_exists(viz.x):
                    cols.append(viz.x)

            elif viz.chart_type == "table":
                # Table: all group_by columns go on the Rows shelf
                if viz.group_by:
                    for col_name in viz.group_by:
                        if self._column_exists(col_name):
                            rows.append(col_name)

            elif viz.chart_type in ("scatter",):
                # Scatter: both axes are measures
                if viz.x and self._column_exists(viz.x):
                    cols.append(viz.x)
                if viz.y and self._column_exists(viz.y):
                    rows.append(viz.y)

            else:
                # Default for bar, line, box: x on Columns, y on Rows
                if viz.x and self._column_exists(viz.x):
                    cols.append(viz.x)
                if viz.y and self._column_exists(viz.y):
                    rows.append(viz.y)

            # If there's a group_by and we haven't used it yet,
            # put the first group column on color for visual separation
            if viz.group_by and not color_field:
                for gb in viz.group_by:
                    if self._column_exists(gb):
                        color_field = gb
                        break

            ws = WorksheetSpec(
                name=ws_name,
                mark_type=mark,
                rows_shelf=rows,
                columns_shelf=cols,
                color_field=color_field,
                size_field=size_field,
                tooltip_fields=tooltip_fields,
                aggregation=agg,
            )
            worksheets.append(ws)

        # Fallback: inject time-series charts if date columns were
        #detected but the recommender didn't include any line charts.
        #This is critical for datasets like Superstore where dates
        #are stored as DD/MM/YYYY strings and get misclassified as text.
        has_line_chart = any(
            w.mark_type == "Line" for w in worksheets
        )
        if not has_line_chart and self._detected_date_cols:
            # find numeric columns suitable for trending
            numeric_cols = self.dataset_schema.numeric_columns or []
            # filter out ID-like columns (Row ID, Postal Code, etc.)
            trend_cols = [
                c for c in numeric_cols
                if not any(kw in c.lower() for kw in ("id", "code", "zip", "postal"))
            ]

            date_col = self._detected_date_cols[0]
            for num_col in trend_cols[:2]:
                ts_name = f"{num_col} over time"
                if ts_name not in seen_names:
                    seen_names.add(ts_name)
                    ts_ws = WorksheetSpec(
                        name=ts_name,
                        mark_type="Line",
                        rows_shelf=[num_col],
                        columns_shelf=[date_col],
                        aggregation="SUM",
                    )
                    # insert at the beginning — time series is usually
                    # the most important chart on a dashboard
                    worksheets.insert(0, ts_ws)
                    logger.info(
                        f"Injected time-series: '{ts_name}' "
                        f"({date_col} × {num_col})"
                    )

        # use the friendly dataset name for the workbook title
        dashboard_title = self._friendly_name + " Dashboard"

        spec = WorkbookSpec(
            name=dashboard_title,
            datasource_name="primary_data",
            worksheets=worksheets,
            dashboard_width=DEFAULT_DASHBOARD_WIDTH,
            dashboard_height=DEFAULT_DASHBOARD_HEIGHT,
        )

        logger.info(f"Planned {len(worksheets)} worksheets.")
        self._workbook_spec = spec
        return spec

    # Stage 2: Build the Tableau XML tree
    

    def _build_xml(self) -> ET.ElementTree:
        """Bringstogether the complete workbook XML structure.

        Follows the exact element ordering found in Tableau-generated .twb files:
          <workbook>
            <document-format-change-manifest>
            <preferences>
            <datasources>
            <worksheets>
            <dashboards>       ← only if multiple worksheets are there
            <windows>          ← required: tells Tableau what to display
          </workbook>
        """
        if self._workbook_spec is None:
            self._plan_worksheets()

        spec = self._workbook_spec

        # Generate the federated datasource name (Tableau convention)
        self._ds_name = f"federated.{uuid4().hex[:24]}"

        # Set up the root workbook tag with version info for build 2026.1
        root = ET.Element("workbook")
        root.set("original-version", "18.1")
        root.set("source-build", TABLEAU_BUILD)
        root.set("source-platform", "win")
        root.set("version", "18.1")
        root.set("xmlns:user", "http://www.tableausoftware.com/xml/user")

        # Modern Tableau builds require this manifest to enable 
        # features like the logical object model (the "Noodles").
        manifest = ET.SubElement(root, "document-format-change-manifest")
        features = [
            "AnimationOnByDefault",
            "MarkAnimation",
            "ObjectModelEncapsulateLegacy",
            "ObjectModelExtractV2",
            "ObjectModelTableType",
            "SchemaViewerObjectModel",
            "SheetIdentifierTracking",
            "_.fcp.VConnDownstreamExtractsWithWarnings.true...VConnDownstreamExtractsWithWarnings",
            "WindowsPersistSimpleIdentifiers"
        ]
        for feature in features:
            ET.SubElement(manifest, feature)

        # Preferences  UI  defaults 
        prefs = ET.SubElement(root, "preferences")
        p1 = ET.SubElement(prefs, "preference")
        p1.set("name", "ui.encoding.shelf.height")
        p1.set("value", "24")
        p2 = ET.SubElement(prefs, "preference")
        p2.set("name", "ui.shelf.height")
        p2.set("value", "26")

        # Generate the data source
        datasources_el = ET.SubElement(root, "datasources")
        self._build_datasource_xml(datasources_el, self._ds_name)

        # Construct each worksheet
        worksheets_el = ET.SubElement(root, "worksheets")
        for ws in spec.worksheets:
            self._build_worksheet_xml(worksheets_el, ws, self._ds_name)

        # Only add a dashboard if there are multiple worksheets
        if len(spec.worksheets) > 1:
            dashboards_el = ET.SubElement(root, "dashboards")
            self._build_dashboard_xml(
                dashboards_el,
                spec.name,
                spec.worksheets,
                spec.dashboard_width,
                spec.dashboard_height,
            )

        # Windows section — tells Tableau which sheet to show on open
        self._build_windows_xml(root, spec.worksheets)

        self._xml_tree = ET.ElementTree(root)
        logger.info("Workbook XML structure assembled successfully.")
        return self._xml_tree

    # --- Datasource XML generation ---
    
    def _build_datasource_xml(self, parent: ET.Element, ds_name: str) -> ET.Element:
        """Builds a portable datasource matching Tableau's native XML structure.

        The structure must be exactly (based on a manually createed .twb file):
          <datasource caption="..." inline="true" name="federated.xxx" version="18.1">
            <connection class="federated">
              <named-connections>
                <named-connection caption="train.csv" name="textscan.xxx">
                  <connection class="textscan" directory="." filename="train.csv" password="" server="" />
                </named-connection>
              </named-connections>
              <relation connection="textscan.xxx" name="train.csv" table="[train#csv]" type="table">
                <columns character-set="UTF-8" header="yes" locale="en_US" separator=",">
                  <column datatype="..." name="..." ordinal="..." />
                </columns>
              </relation>
              <metadata-records>
                <metadata-record class="column">...</metadata-record>
              </metadata-records>
            </connection>
            <aliases enabled="yes" />
            <column ... />   ← field role definitions (dimension/measure)
            <layout dim-ordering="alphabetic" ... />
            <semantic-values>...</semantic-values>
          </datasource>
        """
        ds_el = ET.SubElement(parent, "datasource")
        ds_el.set("caption", self._friendly_name)
        ds_el.set("inline", "true")
        ds_el.set("name", ds_name)
        ds_el.set("version", TABLEAU_VERSION)

        data_filename = Path(self.dataset_schema.source_filename or "train.csv").name
        object_id = f"{data_filename}_{uuid4().hex[:32].upper()}"
        textscan_name = f"textscan.{uuid4().hex[:24]}"

        # ── Connection Layer ──
        conn = ET.SubElement(ds_el, "connection")
        conn.set("class", "federated")

        named_conns = ET.SubElement(conn, "named-connections")
        named_connection = ET.SubElement(named_conns, "named-connection")
        named_connection.set("caption", data_filename)
        named_connection.set("name", textscan_name)

        inner_conn = ET.SubElement(named_connection, "connection")
        inner_conn.set("class", "textscan")
        inner_conn.set("directory", ".")
        inner_conn.set("filename", data_filename)
        inner_conn.set("password", "")
        inner_conn.set("server", "")

        # ── Relation with column definitions ──
        relation = ET.SubElement(conn, "relation")
        relation.set("connection", textscan_name)
        relation.set("name", data_filename)
        relation.set("table", f"[{data_filename.replace('.', '#')}]")
        relation.set("type", "table")

        columns_el = ET.SubElement(relation, "columns")
        columns_el.set("character-set", "UTF-8")
        columns_el.set("header", "yes")
        columns_el.set("locale", "en_US")
        columns_el.set("separator", ",")

        for i, col_schema in enumerate(self.dataset_schema.columns):
            # CRITICAL: Use _col_index for the semantic type, NOT col_schema.semantic_type.
            # _detect_date_columns_fallback() updates _col_index but not dataset_schema.columns.
            # If we use col_schema directly, date columns detected by fallback would be
            # defined as "string" in the datasource XML.
            effective_type = self._col_index[col_schema.name].semantic_type if col_schema.name in self._col_index else col_schema.semantic_type
            tab_info = SEMANTIC_TO_TABLEAU.get(
                effective_type, SEMANTIC_TO_TABLEAU["unknown"]
            )
            col_el = ET.SubElement(columns_el, "column")
            col_el.set("datatype", tab_info["datatype"])
            col_el.set("name", col_schema.name)
            col_el.set("ordinal", str(i))

        # Metadata Records 
        meta_records = ET.SubElement(conn, "metadata-records")
        for i, col_schema in enumerate(self.dataset_schema.columns):
            effective_type = self._col_index[col_schema.name].semantic_type if col_schema.name in self._col_index else col_schema.semantic_type
            tab_info = SEMANTIC_TO_TABLEAU.get(
                effective_type, SEMANTIC_TO_TABLEAU["unknown"]
            )
            col_rec = ET.SubElement(meta_records, "metadata-record")
            col_rec.set("class", "column")

            ET.SubElement(col_rec, "remote-name").text = col_schema.name

            # Remote-type mapping (Tableau internal type codes):
            # 20 = integer, 5 = real/float, 129 = string, 133 = date/datetime
            # These come from inspecting Tableau-generated .twb files manually.
            rtype = "129"
            if tab_info["datatype"] == "real":
                rtype = "5"
            elif tab_info["datatype"] == "integer":
                rtype = "20"
            elif tab_info["datatype"] in ("date", "datetime"):
                rtype = "133"

            ET.SubElement(col_rec, "remote-type").text = rtype
            ET.SubElement(col_rec, "local-name").text = f"[{col_schema.name}]"
            ET.SubElement(col_rec, "parent-name").text = f"[{data_filename}]"
            ET.SubElement(col_rec, "remote-alias").text = col_schema.name
            ET.SubElement(col_rec, "ordinal").text = str(i)
            ET.SubElement(col_rec, "local-type").text = tab_info["datatype"]

            # Aggregation default: Sum for numeric, Count for string, Year for dates
            if tab_info["datatype"] in ("real", "integer"):
                ET.SubElement(col_rec, "aggregation").text = "Sum"
            elif tab_info["datatype"] in ("date", "datetime"):
                ET.SubElement(col_rec, "aggregation").text = "Year"
            else:
                ET.SubElement(col_rec, "aggregation").text = "Count"

            ET.SubElement(col_rec, "contains-null").text = "true"
            ET.SubElement(col_rec, "object-id").text = f"[{object_id}]"

        # Aliases (required after connection block) ──
        aliases = ET.SubElement(ds_el, "aliases")
        aliases.set("enabled", "yes")

        # Column role definitions
        # These <column> elements under <datasource> (outside <connection>)
        # define the dimension/measure roles that Tableau needs.
        for col_schema in self.dataset_schema.columns:
            effective_type = self._col_index[col_schema.name].semantic_type if col_schema.name in self._col_index else col_schema.semantic_type
            tab_info = SEMANTIC_TO_TABLEAU.get(
                effective_type, SEMANTIC_TO_TABLEAU["unknown"]
            )
            role = tab_info["role"]
            col_type = tab_info["type"]
            dt = tab_info["datatype"]

            # Only emit <column> definitions for fields that need
            # non-default settings (dimensions that are numeric, measures, etc.)
            col_def = ET.SubElement(ds_el, "column")
            col_def.set("datatype", dt)
            col_def.set("name", f"[{col_schema.name}]")
            col_def.set("role", role)
            col_def.set("type", col_type)

        # Extract 
        # The extract element is added dynamically by _add_extract_to_datasource()
        # only when export_twbx() is called, because it needs a real .hyper file.
        # Declaring an extract without an actual .hyper file causes Error 51AA5D56.

        # Layout
        layout_el = ET.SubElement(ds_el, "layout")
        layout_el.set("dim-ordering", "alphabetic")
        layout_el.set("measure-ordering", "alphabetic")
        layout_el.set("show-structure", "true")

        # Semantic Values
        sv_el = ET.SubElement(ds_el, "semantic-values")
        sv_val = ET.SubElement(sv_el, "semantic-value")
        sv_val.set("key", "[Country].[Name]")
        sv_val.set("value", '"United States"')

        return ds_el

    # Worksheet XML generation
    def _build_worksheet_xml(self, parent: ET.Element, ws: WorksheetSpec, datasource_name: str) -> ET.Element:
        """Constructs a worksheet following the Tableau XML content model.

        Based on the documented Tableau XML structure (github.com/ranvithm/tableau.xml),
        worksheets with populated shelves MUST have:

            <worksheet name="...">
              <table>
                <view>
                  <datasources>
                    <datasource name="..." />
                  </datasources>
                  <datasource-dependencies datasource="...">
                    <column datatype="string" name="[Category]" role="dimension" type="nominal" />
                    <column-instance column="[Category]" derivation="None" name="[none:Category:nk]" pivot="key" type="nominal" />
                    ...
                  </datasource-dependencies>
                  <aggregation value="true" />
                </view>
                <style />
                <panes>...</panes>
                <rows>[ds].[none:Category:nk]</rows>
                <cols>[ds].[sum:Sales:qk]</cols>
              </table>
              <simple-id uuid="..." />
            </worksheet>

        The <datasource-dependencies> block with both <column> and <column-instance>
        elements is REQUIRED for Tableau to resolve the shelf references.
        The column-instance name attribute MUST be bracket-wrapped: [none:Field:nk]
        """
        ws_el = ET.SubElement(parent, "worksheet")
        ws_el.set("name", ws.name)

        table_el = ET.SubElement(ws_el, "table")

        # <view> inside <table>
        view_el = ET.SubElement(table_el, "view")
        datasources_el = ET.SubElement(view_el, "datasources")
        ds_ref = ET.SubElement(datasources_el, "datasource")
        ds_ref.set("caption", self._friendly_name)
        ds_ref.set("name", datasource_name)

        # <datasource-dependencies> — register all fields used on shelves
        all_fields = set(ws.rows_shelf + ws.columns_shelf)
        if ws.color_field:
            all_fields.add(ws.color_field)
        if ws.size_field:
            all_fields.add(ws.size_field)

        if all_fields:
            deps = ET.SubElement(view_el, "datasource-dependencies")
            deps.set("datasource", datasource_name)

            for field in sorted(all_fields):
                col_schema = self._col_index.get(field)
                if col_schema is None and field != "Number of Records":
                    continue

                effective_type = col_schema.semantic_type if col_schema else "numeric"
                tab_info = SEMANTIC_TO_TABLEAU.get(effective_type, SEMANTIC_TO_TABLEAU["unknown"])

                role = tab_info["role"]
                col_type = tab_info["type"]
                dt = tab_info["datatype"]

                # <column> element — declares the field
                col_el = ET.SubElement(deps, "column")
                col_el.set("datatype", dt)
                col_el.set("name", f"[{field}]")
                col_el.set("role", role)
                col_el.set("type", col_type)

                # <column-instance> element — declares the shelf reference
                col_inst = ET.SubElement(deps, "column-instance")
                col_inst.set("column", f"[{field}]")

                if role == "measure":
                    derivation = "Sum"
                    agg_lower = (ws.aggregation or "SUM").lower()
                    type_key = "qk"
                    inst_name = f"[{agg_lower}:{field}:{type_key}]"
                    inst_type = "quantitative"
                else:
                    derivation = "None"
                    if effective_type == "datetime":
                        type_key = "ok"
                    else:
                        type_key = "nk"
                    inst_name = f"[none:{field}:{type_key}]"
                    inst_type = "nominal" if type_key == "nk" else "ordinal"

                col_inst.set("derivation", derivation)
                col_inst.set("name", inst_name)  # MUST be bracket-wrapped
                col_inst.set("pivot", "key")
                col_inst.set("type", inst_type)

        # <aggregation> is required inside <view>
        agg_el = ET.SubElement(view_el, "aggregation")
        agg_el.set("value", "true")

        # <style>
        ET.SubElement(table_el, "style")

        # <panes>
        panes_el = ET.SubElement(table_el, "panes")
        pane_el = ET.SubElement(panes_el, "pane")
        pane_el.set("selection-relaxation-option", "selection-relaxation-allow")
        pane_view = ET.SubElement(pane_el, "view")
        ET.SubElement(pane_view, "breakdown").set("value", "auto")
        ET.SubElement(pane_el, "mark").set("class", ws.mark_type or "Automatic")

        # Encodings — color and size shelves (needed for pie/heatmap)
        if ws.color_field or ws.size_field:
            encodings_el = ET.SubElement(pane_el, "encodings")
            if ws.color_field:
                color_el = ET.SubElement(encodings_el, "color")
                color_el.set(
                    "column",
                    self._build_column_instance_ref(
                        ws.color_field, datasource_name, ws.aggregation,
                    ),
                )
            if ws.size_field:
                size_el = ET.SubElement(encodings_el, "size")
                size_el.set(
                    "column",
                    self._build_column_instance_ref(
                        ws.size_field, datasource_name, ws.aggregation,
                    ),
                )

        # <rows> and <cols>
        rows_el = ET.SubElement(table_el, "rows")
        if ws.rows_shelf:
            refs = [self._build_column_instance_ref(f, datasource_name, ws.aggregation) for f in ws.rows_shelf]
            rows_el.text = f"({' / '.join(refs)})" if len(refs) > 1 else refs[0]

        cols_el = ET.SubElement(table_el, "cols")
        if ws.columns_shelf:
            refs = [self._build_column_instance_ref(f, datasource_name, ws.aggregation) for f in ws.columns_shelf]
            cols_el.text = " / ".join(refs)

        # <simple-id>
        simple_id = ET.SubElement(ws_el, "simple-id")
        simple_id.set("uuid", f"{{{str(uuid4()).upper()}}}")

        return ws_el

    def _add_column_dep(self, parent: ET.Element, field_name: str):
        """Helper to add column-instance dependencies required by the Gold File."""
        # This mimics the <column-instance> tags seen in Gold File 
        col_inst = ET.SubElement(parent, "column-instance")
        col_inst.set("column", f"[{field_name}]")
        col_inst.set("derivation", "None" if self._resolve_column_role(field_name) == "dimension" else "Sum")
        col_inst.set("name", self._build_column_instance_ref(field_name, "", "").split('.')[-1].strip('[]'))
        col_inst.set("pivot", "key")
        col_inst.set("type", "nominal" if self._resolve_column_role(field_name) == "dimension" else "quantitative")
    # --- Dashboard XML generation ---
    def _build_dashboard_xml(self, parent: ET.Element, title: str, worksheets: List[WorksheetSpec], width: int, height: int) -> ET.Element:
        """Constructs a dashboard with the correct element sequence."""
        dash_el = ET.SubElement(parent, "dashboard")
        dash_el.set("name", title)

        ET.SubElement(dash_el, "style")

        size_el = ET.SubElement(dash_el, "size")
        size_el.set("maxheight", str(height))
        size_el.set("maxwidth", str(width))
        size_el.set("minheight", str(height))
        size_el.set("minwidth", str(width))

        zones_el = ET.SubElement(dash_el, "zones")
        for ws in worksheets:
            self._add_worksheet_zone(zones_el, ws.name, str(width), str(height))

        simple_id = ET.SubElement(dash_el, "simple-id")
        simple_id.set("uuid", f"{{{str(uuid4()).upper()}}}")

        return dash_el
    
    def _add_worksheet_zone(self, parent, sheet_name, w, h):
        """Add a zone element that embeds a worksheet into the dashboard."""
        zone = ET.SubElement(parent, "zone")
        zone.set("h", str(h))
        zone.set("id", str(self._next_zone_id()))
        zone.set("name", sheet_name)
        zone.set("w", str(w))
        zone.set("x", "0")
        zone.set("y", "0")
        return zone

    def _build_windows_xml(self, root: ET.Element, worksheets: List[WorksheetSpec]) -> None:
        """Build <windows> — tells Tableau which sheet to show on open."""
        windows_el = ET.SubElement(root, "windows")
        windows_el.set("saved-dpi-scale-factor", "1.25")
        windows_el.set("source-height", "37")

        if not worksheets:
            return

        first_ws = worksheets[0]
        window = ET.SubElement(windows_el, "window")
        window.set("class", "worksheet")
        window.set("maximized", "true")
        window.set("name", first_ws.name)

        cards = ET.SubElement(window, "cards")

        left_edge = ET.SubElement(cards, "edge")
        left_edge.set("name", "left")
        left_strip = ET.SubElement(left_edge, "strip")
        left_strip.set("size", "160")
        for card_type in ("pages", "filters", "marks"):
            card = ET.SubElement(left_strip, "card")
            card.set("type", card_type)

        top_edge = ET.SubElement(cards, "edge")
        top_edge.set("name", "top")
        for card_type, size in [("columns", "2147483647"), ("rows", "2147483647"), ("title", "30")]:
            strip = ET.SubElement(top_edge, "strip")
            strip.set("size", size)
            card = ET.SubElement(strip, "card")
            card.set("type", card_type)

        simple_id = ET.SubElement(window, "simple-id")
        simple_id.set("uuid", f"{{{str(uuid4()).upper()}}}")

    # Stage 3: Export to .twb or .twbx

    def export_twb(self, output_path: str) -> str:
        """Write the workbook as a plain .twb (XML) file.

        Args:
            output_path: Where to save the file.  Parent directories
                         are created automatically.
        Returns:
            The absolute path to the written file.
        """
        if self._xml_tree is None:
            self._build_xml()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        # ElementTree.write() produces flat XML by default.
        # We manually indent it first so the output is human-readable
        # and diffs cleanly in git — makes debugging much easier.
        self._indent_xml(self._xml_tree.getroot())

        self._xml_tree.write(
            str(out),
            encoding="utf-8",
            xml_declaration=True,
        )

        file_size = out.stat().st_size
        logger.info(
            f"Exported .twb → {out.resolve()} "
            f"({file_size / 1024:.1f} KB)"
        )
        return str(out.resolve())

    def export_twbx(
        self,
        output_path: str,
        data_file: Optional[str] = None,
    ) -> str:
        """Write the workbook as a packaged .twbx (ZIP) file with a Hyper extract.

        A .twbx is a ZIP archive containing:
          - the .twb XML file
          - a Data/ folder with the .hyper extract
          - the source CSV file (as fallback)

        The Hyper extract is required for Tableau Public compatibility.
        We use pantab to generate the .hyper file from the DataFrame.

        Args:
            output_path: Where to save the .twbx.
            data_file:   Path to the data file to bundle.  If None, we
                         export the DataFrame as a CSV into the archive.
        Returns:
            The absolute path to the written file.
        """
        if self._xml_tree is None:
            self._build_xml()

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        hyper_filename = "Data/Extracts/Extract.hyper"

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate the .hyper extract from the DataFrame
            hyper_path = os.path.join(tmpdir, "Extract.hyper")
            self._generate_hyper_extract(hyper_path)

            # Add the <extract> element into the datasource XML
            # pointing to the .hyper file inside the archive
            self._inject_extract_element(hyper_filename)

            # Write the .twb
            twb_name = out.stem + ".twb"
            twb_path = os.path.join(tmpdir, twb_name)
            self._indent_xml(self._xml_tree.getroot())
            self._xml_tree.write(twb_path, encoding="utf-8", xml_declaration=True)

            # Add the CSV data file
            if data_file and Path(data_file).exists():
                data_src = str(Path(data_file).resolve())
                data_filename = Path(data_file).name
            else:
                data_filename = (
                    self.dataset_schema.source_filename or "data.csv"
                )
                data_src = os.path.join(tmpdir, data_filename)
                self.df.to_csv(data_src, index=False, encoding="utf-8")

            # Create the zip archive
            with zipfile.ZipFile(str(out), "w", zipfile.ZIP_DEFLATED) as zf:
                zf.write(twb_path, twb_name)
                zf.write(data_src, data_filename)
                zf.write(hyper_path, hyper_filename)

        # CRITICAL: Remove the <extract> element from the in-memory XML tree
        # after writing the .twbx. If export_twb() is called later, it must
        # NOT have the extract reference (since .twb files don't bundle data).
        # Without this cleanup, the .twb would reference a non-existent .hyper
        # file and Tableau would crash on open.
        self._remove_extract_element()

        file_size = out.stat().st_size
        logger.info(
            f"Exported .twbx → {out.resolve()} "
            f"({file_size / 1024:.1f} KB, with Hyper extract)"
        )
        return str(out.resolve())

    def _generate_hyper_extract(self, hyper_path: str) -> None:
        """Generate a .hyper extract file from the DataFrame using pantab.

        pantab wraos the Tableau's Hyper API and handles all type conversions
        automatically.  The table is written to the Extract.Extract path
        (schema="Extract", table="Extract") which is Tableau's default
        extract location.

        IMPORTANT: Using table="Extract" (simple string) creates the table
        at public.Extract, but Tableau looks for Extract.Extract. We must
        use TableName("Extract", "Extract") to avoid any errors while opening the .twbx in Tableau Desktop or Public.
        """
        try:
            import pantab
            from tableauhyperapi import TableName

            pantab.frame_to_hyper(
                self.df,
                hyper_path,
                table=TableName("Extract", "Extract"),
            )
            logger.info(
                f"Generated Hyper extract: {hyper_path} "
                f"({self.df.shape[0]} rows × {self.df.shape[1]} cols)"
            )
        except ImportError:
            logger.warning(
                "pantab not installed — cannot generate .hyper extract. "
                "Install with: pip install pantab"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to generate Hyper extract: {e}")
            raise

    def _inject_extract_element(self, hyper_filename: str) -> None:
        """Inject an <extract> element into the datasource XML.

        This must be called AFTER _build_xml() and BEFORE writing the .twb.
        The element is inserted between <column> definitions and <layout>,
        matching the structure in Tableau-generated .twb files.

        The hyper_filename is the path INSIDE the .twbx archive, e.g.
        'Data/Extracts/Extract.hyper'.
        """
        root = self._xml_tree.getroot()
        ds_el = root.find(".//datasource[@inline='true']")
        if ds_el is None:
            logger.warning("No inline datasource found — cannot inject extract")
            return

        # Find the <layout> element to insert <extract> before it
        layout_el = ds_el.find("layout")
        if layout_el is not None:
            insert_idx = list(ds_el).index(layout_el)
        else:
            insert_idx = len(list(ds_el))

        # Build the extract element
        extract_el = ET.Element("extract")
        extract_el.set("_.fcp.VConnDownstreamExtractsWithWarnings.true...user-specific", "false")
        extract_el.set("count", "-1")
        extract_el.set("enabled", "true")
        extract_el.set("object-id", "")
        extract_el.set("units", "records")

        ext_conn = ET.SubElement(extract_el, "connection")
        ext_conn.set("access_mode", "readonly")
        ext_conn.set("author-locale", "en_US")
        ext_conn.set("class", "hyper")
        ext_conn.set("dbname", hyper_filename)
        ext_conn.set("default-settings", "hyper")
        ext_conn.set("schema", "Extract")
        ext_conn.set("sslmode", "")
        ext_conn.set("tablename", "Extract")
        ext_conn.set("username", "tableau_internal_user")

        ext_relation = ET.SubElement(ext_conn, "relation")
        ext_relation.set("name", "Extract")
        ext_relation.set("table", "[Extract].[Extract]")
        ext_relation.set("type", "table")

        # Insert before <layout>
        ds_el.insert(insert_idx, extract_el)
        logger.info(f"Injected extract element pointing to {hyper_filename}")

    def _remove_extract_element(self) -> None:
        """Remove the <extract> element from the datasource XML.

        Called after export_twbx() to ensure the in-memory XML tree
        is clean for any subsequent export_twb() calls.
        """
        root = self._xml_tree.getroot()
        ds_el = root.find(".//datasource[@inline='true']")
        if ds_el is None:
            return
        extract_el = ds_el.find("extract")
        if extract_el is not None:
            ds_el.remove(extract_el)
            logger.info("Removed extract element from in-memory XML tree")

    # Validation


    def validate(self) -> Tuple[bool, List[str]]:
        """Check the generated workbook for common problems.

        Runs a series of basic checks on the XML tree and returns
        a tuple of (is_valid, list_of_issues).  An empty issues list
        means everything looks good.

        This is not a full Tableau XML schema validation (Tableau
        doesn't publish an XSD), but it checks the mistakes with a file 
        we created manually in Tableau Desktop so avoid errorrs.
        """
        issues = []

        if self._xml_tree is None:
            try:
                self._build_xml()
            except Exception as e:
                issues.append(f"Failed to build XML: {e}")
                return False, issues

        root = self._xml_tree.getroot()

        # check 1: root tag should be "workbook"
        if root.tag != "workbook":
            issues.append(
                f"Root element is '{root.tag}', expected 'workbook'."
            )

        # check 2: version attribute should be present
        if not root.get("version"):
            issues.append("Missing 'version' attribute on <workbook>.")

        # check 3: at least one datasource
        ds_elements = root.findall(".//datasource")
        if not ds_elements:
            issues.append("No <datasource> elements found.")

        # check 4: at least one worksheet
        ws_elements = root.findall(".//worksheet")
        if not ws_elements:
            issues.append("No <worksheet> elements found.")
        else:
            # check that every worksheet has a name
            for ws_el in ws_elements:
                if not ws_el.get("name"):
                    issues.append("Found a <worksheet> without a 'name' attribute.")

        # check 5: at least one dashboard
        dash_elements = root.findall(".//dashboard")
        if not dash_elements:
            issues.append("No <dashboard> elements found.")

        # check 6: all columns referenced on shelves actually exist
        # in the datasource
        known_cols = {
            col.name for col in self.dataset_schema.columns
        }
        known_cols.add("Number of Records")  # our calculated field

        if self._workbook_spec:
            for ws in self._workbook_spec.worksheets:
                for field in ws.rows_shelf + ws.columns_shelf:
                    if field not in known_cols:
                        issues.append(
                            f"Worksheet '{ws.name}' references unknown "
                            f"column '{field}'."
                        )
                if ws.color_field and ws.color_field not in known_cols:
                    issues.append(
                        f"Worksheet '{ws.name}' color field "
                        f"'{ws.color_field}' not in dataset."
                    )
                if ws.size_field and ws.size_field not in known_cols:
                    issues.append(
                        f"Worksheet '{ws.name}' size field "
                        f"'{ws.size_field}' not in dataset."
                    )

        # check 7: dashboard zones reference valid worksheet names
        # Worksheet zones are identified by having a 'name' attribute
        # (without type-v2).  Layout zones have type-v2 but no name.
        ws_names = {ws_el.get("name") for ws_el in ws_elements}
        for zone in root.iter("zone"):
            zname = zone.get("name")
            zone_type = zone.get("type-v2")
            # only check zones that reference a worksheet (have name,
            # no type-v2 — or type-v2 is not a layout type)
            if zname and zone_type is None:
                if zname not in ws_names:
                    issues.append(
                        f"Dashboard zone references worksheet '{zname}' "
                        f"which doesn't exist."
                    )

        is_valid = len(issues) == 0
        if is_valid:
            logger.info("Validation passed — no issues found.")
        else:
            logger.warning(
                f"Validation found {len(issues)} issue(s): "
                + "; ".join(issues[:3])
            )
        return is_valid, issues

    # Calculated fields — public method for adding custom ones

    def add_calculated_field(
        self,
        field_name: str,
        formula: str,
        datatype: str = "real",
        role: str = "measure",
    ) -> None:
        """Add a custom calculated field to the datasource.

        This should be called BEFORE export_twb/export_twbx. If the
        XML has already been built, we rebuild it to include the new field.

        Args:
            field_name: Display name (e.g. "Profit Ratio").
            formula:    Tableau formula string (e.g. "SUM([Profit])/SUM([Sales])").
            datatype:   One of "real", "integer", "string", "boolean", "datetime".
            role:       "measure" or "dimension".

        Example:
            generator.add_calculated_field(
                "Profit Ratio",
                "SUM([Profit]) / SUM([Sales])",
            )
        """
        # If the tree was already built, force a rebuild so the new
        # field shows up in the XML — can't patch an existing tree easily
        if self._xml_tree is not None:
            logger.info(
                f"XML was already built — will rebuild after adding "
                f"calculated field '{field_name}'."
            )
            self._xml_tree = None

        # We store the calculated fields and inject them during
        # _build_datasource_xml when the XML tree gets assembled.
        self._extra_calc_fields.append({
            "name": field_name,
            "formula": formula,
            "datatype": datatype,
            "role": role,
        })

        # Also register the field name so validation doesn't flag it
        # as unknown when it appears on a shelf later
        dummy_schema = ColumnSchema(
            name=field_name,
            semantic_type="numeric" if role == "measure" else "categorical",
        )
        self._col_index[field_name] = dummy_schema

        logger.info(f"Queued calculated field: {field_name} = {formula}")

    # Private helpers
    
    # Common keywords that usually indicate a date/time column.
    # Mirrors the approach in DashboardAnalyzer._DATE_KW.
    _DATE_KEYWORDS = {
        "date", "time", "datetime", "timestamp", "year", "month",
        "day", "week", "quarter", "period", "created", "updated",
        "modified", "registered", "expired", "due", "start", "end",
    }

    def _detect_date_columns_fallback(self) -> List[str]:
        """Scan the DataFrame for columns that look like dates but weren't
        detected by the upstream schema.

        Some datasets store dates as strings in DD/MM/YYYY or other
        ambiguous formats that detect_semantic_type can miss.
        We catch them here by checking column names against known date
        keywords and then trying to parse a small sample.

        Returns a list of column names that are likely date columns.
        """
        already_datetime = set(self.dataset_schema.datetime_columns or [])
        detected = []

        for col in self.df.columns:
            # skip columns already identified as datetime
            if col in already_datetime:
                continue

            # check if the column name contains a date keyword
            col_lower = col.lower().replace("_", " ")
            name_looks_like_date = any(
                kw in col_lower for kw in self._DATE_KEYWORDS
            )
            if not name_looks_like_date:
                continue

            # try parsing a sample — if >80% parse successfully, treat
            # it as a date column
            sample = self.df[col].dropna().head(200)
            if len(sample) == 0:
                continue

            try:
                parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
                success_ratio = parsed.notna().mean()
                if success_ratio >= 0.8:
                    detected.append(col)
                    # NOTE: We intentionally do NOT update self._col_index here.
                    # The CSV stores these as strings, and Tableau's textscan
                    # driver will read them as strings. If we declare them as
                    # "datetime" in the datasource XML, shelf references will
                    # use ":ok" (ordinal/date).
                    #
                    # This list is only used for worksheet planning (adding
                    # time-series line charts). The shelf references will use
                    # ":nk" (nominal/string) which matches the actual CSV type.
                    logger.info(
                        f"Date fallback: '{col}' looks like a date "
                        f"({success_ratio:.0%} parse success) — will use for "
                        f"time-series charts but keeping string type in datasource"
                    )
            except Exception:
                pass

        return detected

    _zone_counter = 0  # class-level counter for unique zone IDs

    def _next_zone_id(self) -> int:
        """Generate a unique integer ID for dashboard zone elements."""
        TableauWorkbookGenerator._zone_counter += 1
        return TableauWorkbookGenerator._zone_counter

    def _column_exists(self, col_name: str) -> bool:
        """Check whether a column name is in the dataset or is a known
        calculated field like 'Number of Records'."""
        if col_name in self._col_index:
            return True
        if col_name == "Number of Records":
            return True
        logger.warning(f"Column '{col_name}' not found in dataset — skipping.")
        return False

    def _resolve_column_role(self, col_name: str) -> str:
        """Determine if a column should be treated as a dimension or measure
        in the context of a Tableau shelf expression.

        Numeric columns that appear on the x-axis of a bar chart should
        sometimes be dimensions (e.g. a "Year" column).  We check the
        DatasetSchema's semantic type to decide.

        Special case: "Number of Records" is always a measure because
        it's our built-in calculated field (formula = 1, aggregated as SUM).
        """
        # calculated field we always create — it's inherently a measure
        if col_name == "Number of Records":
            return "measure"

        schema = self._col_index.get(col_name)
        if schema is None:
            return "dimension"
        if schema.semantic_type in ("numeric",):
            return "measure"
        return "dimension"

    def _build_column_instance_ref(
        self,
        field_name: str,
        datasource_name: str,
        aggregation: Optional[str] = None,
    ) -> str:
        """Build the column-instance reference for rows/cols/encodings.

        Tableau uses a specific naming format for field references:
            [datasource].[agg:FieldName:typekey]

        Where:
          - agg is the aggregation in lowercase (sum, avg, none)
          - FieldName is the column name
          - typekey is: nk (nominal/string), ok (ordinal/date),
                        qk (quantitative/numeric)

        Examples:
          [primary_data].[none:Category:nk]     — dimension on shelf
          [primary_data].[sum:Sales:qk]         — SUM(Sales) on shelf
          [primary_data].[none:Order Date:ok]    — date dimension

        "Number of Records" is special — it's a calculated field that
        always uses sum:Number of Records:qk.
        """
        role = self._resolve_column_role(field_name)

        if role == "measure":
            agg_lower = (aggregation or "sum").lower()
            type_key = "qk"
            inst_name = f"[{agg_lower}:{field_name}:{type_key}]"
        else:
            # dimensions — check if it's a date (ordinal) or string (nominal)
            col_schema = self._col_index.get(field_name)
            if col_schema and col_schema.semantic_type == "datetime":
                type_key = "ok"
            else:
                type_key = "nk"
            inst_name = f"[none:{field_name}:{type_key}]"

        return f"[{datasource_name}].{inst_name}"

    def _build_field_expression(
        self,
        field_name: str,
        datasource_name: str,
        aggregation: Optional[str] = None,
    ) -> str:
        """Legacy method — delegates to _build_column_instance_ref.

        Kept for backward compatibility with any code that calls it.
        """
        return self._build_column_instance_ref(
            field_name, datasource_name, aggregation
        )

    @staticmethod
    def _sanitize_sheet_name(raw_name: str) -> str:
        """Remove characters that Tableau doesn't allow in sheet names.

        Tableau rejects slashes, backslashes, square brackets, and a
        few others.  We also truncate to 50 chars to keep things tidy.
        """
        # strip out problematic characters
        cleaned = raw_name
        for ch in r"/\\[]{}:*?<>|\"'":
            cleaned = cleaned.replace(ch, "")
        # collapse multiple spaces
        cleaned = " ".join(cleaned.split())
        # truncate
        if len(cleaned) > 50:
            cleaned = cleaned[:50].rstrip()
        return cleaned or "Sheet"

    @staticmethod
    def _indent_xml(elem: ET.Element, level: int = 0) -> None:
        """Add whitespace indentation to an ElementTree for pretty printing.

        Python 3.9+ has ET.indent() but we do it manually to support
        3.8 and 3.9 alike.  This is a standard recursive approach.
        """
        indent = "\n" + ("  " * level)
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
            for child in elem:
                TableauWorkbookGenerator._indent_xml(child, level + 1)
            # after processing children, fix the last child's tail
            if not child.tail or not child.tail.strip():
                child.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent
            # preserve text content (like shelf expressions)
            # don't overwrite elem.text if it has actual data

# Standalone test — run: python tableau_workbook_generator.py

if __name__ == "__main__":
    import sys
    import json
    from scripts.schemas import (
        build_dataset_schema,
        build_profile_report,
        VisualizationSpec,
        DashboardSpec,
    )
    from scripts.visualization_recommender import VisualizationRecommender

    # You can also test it as a command-line argument as shown below:
    # python tableau_workbook_generator train.csv
    #----------------------
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "train.csv"   # Change this if files is other path

    df = pd.read_csv(file_path)
    print(f"Loaded: {file_path}")

    print(f"  Rows: {len(df)}  |  Columns: {len(df.columns)}")
    print(f"  Columns: {list(df.columns)}\n")

    # build the schema and profile
    ds = build_dataset_schema(file_path, df)
    profile = build_profile_report(ds, df)

    # run the visualization recommender
    recommender = VisualizationRecommender()
    dashboard = recommender.recommend(ds, profile)

    print(f"Dashboard: {dashboard.name}")
    print(f"  Visuals: {len(dashboard.visuals)}")

    #generate the Tableau workbook
    generator = TableauWorkbookGenerator(
        dashboard_spec=dashboard,
        dataset_schema=ds,
        dataframe=df,
    )

    # optionally add a custom calculated field
    if "Sales" in df.columns and "Profit" in df.columns:
        generator.add_calculated_field(
            "Profit Ratio",
            "SUM([Profit]) / SUM([Sales])",
            datatype="real",
            role="measure",
        )

    # validate before exporting
    is_valid, issues = generator.validate()
    print(f"\n  Validation: {'PASSED' if is_valid else 'FAILED'}")
    if issues:
        for issue in issues:
            print(f"    - {issue}")

    # export both formats
    twb_path = generator.export_twb("output/train_dashboard.twb")
    print(f"\n  .twb saved: {twb_path}")

    twbx_path = generator.export_twbx("output/train_dashboard.twbx", data_file=file_path)
    print(f"  .twbx saved: {twbx_path}")

    # print a snippet of the XML so you can see what it looks like for future reference.
    print("\n--- XML Preview (first 60 lines) ---")
    with open(twb_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 60:
                print("  ...")
                break
            print(f"  {line}", end="")

    print("\nDone.")
