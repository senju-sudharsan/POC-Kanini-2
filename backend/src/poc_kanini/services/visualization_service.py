"""Dataset-agnostic visualization engine computing grounded chart specs directly from tabular data."""

import io
import json
import logging
import re
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

ChartType = Literal["bar", "line", "pie", "donut", "scatter", "table", "kpi"]


class KpiCard(BaseModel):
    """Key Performance Indicator summary metric."""

    label: str
    value: str | int | float
    subtext: str | None = None


class VisualizationPayload(BaseModel):
    """Structured, dataset-agnostic visualization specification."""

    chart_type: ChartType = "bar"
    title: str
    description: str = ""
    x_field: str | None = None
    y_field: str | None = None
    series_field: str | None = None
    data: list[dict[str, Any]] = Field(default_factory=list)
    kpis: list[KpiCard] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    error: str | None = None


def _to_dataframe(data: list[dict[str, Any]] | str | pd.DataFrame) -> pd.DataFrame:
    """Normalize input into a clean pandas DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
    if isinstance(data, str):
        cleaned = data.strip()
        if not cleaned:
            return pd.DataFrame()
        try:
            return pd.read_csv(io.StringIO(cleaned))
        except Exception:
            # Fallback to json records if str is json
            try:
                records = json.loads(cleaned)
                if isinstance(records, list):
                    return pd.DataFrame(records)
            except Exception:
                pass
            raise ValueError("Unable to parse string as CSV or JSON tabular data.")
    raise ValueError(f"Unsupported data type: {type(data)}")


class VisualizationService:
    """Computes genuinely data-grounded visualization specifications from arbitrary tabular datasets."""

    def visualize(
        self,
        data: list[dict[str, Any]] | str | pd.DataFrame,
        query: str = "",
        chart_type: str | None = None,
        x_field: str | None = None,
        y_field: str | None = None,
        aggregation: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Generate one or more grounded visualization specifications based on query and dataset schema."""
        try:
            df = _to_dataframe(data)
        except Exception as err:
            return [VisualizationPayload(
                chart_type="table",
                title="Visualization Error",
                error=f"Invalid tabular dataset: {err}",
            ).model_dump()]

        if df.empty or len(df.columns) == 0:
            return [VisualizationPayload(
                chart_type="table",
                title="Empty Dataset",
                error="The dataset is empty. Cannot generate visualizations.",
            ).model_dump()]

        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        q_lower = query.lower().strip()

        # Check for multi-clause chart requests (e.g., "A bar chart of... A line chart of... A scatter chart of...")
        clauses = self._split_multi_chart_clauses(query)
        if len(clauses) >= 2:
            specs = []
            for clause in clauses:
                sub_res = self._visualize_single_clause(df, clause, limit=limit)
                if sub_res:
                    specs.append(sub_res)
            if specs:
                return [s.model_dump() for s in specs]

        # Handle generic multi-visualization requests: e.g., "Give me 5 useful visualizations based on this dataset", "Give me 5 charts", "create 3 visualizations"
        m_multi = re.search(r"\b(?:give\s+me\s+|generate\s+|show\s+|create\s+)?([2-5]|two|three|four|five)\s+(?:[a-z]+\s+)*(?:visualizations?|charts?|plots?|graphs?)\b", q_lower)
        if m_multi and not clauses:
            word_map = {"two": 2, "three": 3, "four": 4, "five": 5}
            token = m_multi.group(1).lower()
            num_charts = word_map.get(token, int(token) if token.isdigit() else 3)
            return self._generate_multi_visualizations(df, num_charts=num_charts, query=query)

        # Single visualization resolution
        spec = self._visualize_single_clause(
            df=df,
            query=query,
            chart_type=chart_type,
            explicit_x=x_field,
            explicit_y=y_field,
            aggregation=aggregation,
            limit=limit,
        )
        return [spec.model_dump()]

    def _split_multi_chart_clauses(self, query: str) -> list[str]:
        """Detect and split individual chart specifications within a multi-chart prompt."""
        if not query:
            return []
        # Split on sentence boundaries, list markers, or newlines
        parts = [s.strip() for s in re.split(r"(?:(?<=[.!?\n])\s+|\n+|(?:^|\s+)(?:[1-9]\.|\*|-)\s+)", query) if s.strip()]
        chart_clauses = []
        _CHART_INDICATORS = (
            "bar chart", "line chart", "scatter chart", "scatter plot", "pie chart",
            "donut chart", "kpi card", "table preview", "chart comparing", "chart showing",
            "plot comparing", "plot showing", "visualization comparing", "histogram",
        )
        for part in parts:
            p_lower = part.lower()
            if any(ind in p_lower for ind in _CHART_INDICATORS):
                chart_clauses.append(part)
            elif re.search(r"\b(?:a\s+)?(?:bar|line|scatter|pie|donut|histogram)\s+(?:chart|plot|graph)\b", p_lower):
                chart_clauses.append(part)

        return chart_clauses

    def _visualize_single_clause(
        self,
        df: pd.DataFrame,
        query: str,
        chart_type: str | None = None,
        explicit_x: str | None = None,
        explicit_y: str | None = None,
        aggregation: str | None = None,
        limit: int | None = None,
    ) -> VisualizationPayload:
        """Process a single chart query against the dataset schema."""
        q_lower = query.lower().strip()

        # Detect or resolve requested columns
        detected_x, detected_y = self._resolve_fields_from_query(df, q_lower, explicit_x, explicit_y)

        # Validate explicitly requested fields that do not exist in the dataset
        missing_fields = self._find_missing_explicit_fields(df, q_lower, explicit_x, explicit_y)
        if missing_fields:
            return VisualizationPayload(
                chart_type="table",
                title="Field Not Found",
                error=f"I cannot create that visualization because the dataset does not contain field(s): {', '.join(missing_fields)}. Available columns: {', '.join(df.columns)}.",
                columns=list(df.columns),
            )

        # Detect chart type if not explicitly supplied
        effective_chart_type = self._resolve_chart_type(df, q_lower, chart_type, detected_x, detected_y)

        # Build visualization based on resolved type
        return self._build_single_visualization(
            df=df,
            chart_type=effective_chart_type,
            x_col=detected_x,
            y_col=detected_y,
            query=query,
            aggregation=aggregation,
            limit=limit,
        )

    def _is_id_column(self, col: str, df: pd.DataFrame) -> bool:
        """Check if a column is an identifier / surrogate key rather than a categorical feature."""
        c = col.lower().strip()
        if c in ("id", "uuid", "guid", "pk", "key", "index"):
            return True
        if re.search(r"(_id|_uuid|_guid|_key|_pk|_code)$", c):
            return True
        # Check high cardinality unique string identifiers
        if not pd.api.types.is_numeric_dtype(df[col]) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            if len(df) > 5 and df[col].nunique() == len(df):
                return True
        return False

    def _extract_syntactic_grouping_field(self, df: pd.DataFrame, q_lower: str) -> str | None:
        """Extract explicitly requested grouping column from phrases like 'by <col>', 'per <col>', etc."""
        col_map = {c.lower(): c for c in df.columns}

        # 1. Match 'by <field>', 'grouped by <field>', 'per <field>', 'across <field>', 'for each <field>'
        patterns = [
            r"\b(?:by|grouped\s+by|group\s+by|grouping\s+by|across|per|for\s+each|breakdown\s+by)\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)?)\b",
            r"\b(?:using|on)\s+([a-zA-Z0-9_]+(?:\s+[a-zA-Z0-9_]+)?)\b",
        ]
        for pat in patterns:
            matches = re.finditer(pat, q_lower)
            for m in matches:
                target = m.group(1).strip()
                # Direct match
                if target in col_map:
                    return col_map[target]
                # Underscore vs space match
                target_under = target.replace(" ", "_")
                if target_under in col_map:
                    return col_map[target_under]
                # Check first word if multi-word: e.g. "order_status aggregated by month" -> "order_status"
                first_word = target.split()[0]
                if first_word in col_map:
                    return col_map[first_word]

        return None

    def _resolve_fields_from_query(
        self,
        df: pd.DataFrame,
        q_lower: str,
        explicit_x: str | None,
        explicit_y: str | None,
    ) -> tuple[str | None, str | None]:
        """Match query tokens against actual dataset column names with priority on syntactic intent and non-ID dimensions."""
        cols = list(df.columns)
        col_map = {c.lower(): c for c in cols}

        resolved_x = col_map.get(explicit_x.lower()) if explicit_x and explicit_x.lower() in col_map else None
        resolved_y = col_map.get(explicit_y.lower()) if explicit_y and explicit_y.lower() in col_map else None

        if resolved_x and resolved_y:
            return resolved_x, resolved_y

        dt_cols = self._detect_datetime_columns(df)
        numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and c not in dt_cols]
        categorical_cols = [c for c in cols if c not in numeric_cols and c not in dt_cols and not self._is_id_column(c, df)]
        id_cols = [c for c in cols if self._is_id_column(c, df) and c not in dt_cols]

        # 1. Syntactic grouping extraction: 'by <col>', 'per <col>', etc.
        syntactic_x = self._extract_syntactic_grouping_field(df, q_lower)
        if syntactic_x and not resolved_x:
            resolved_x = syntactic_x

        # 2. Score candidate column mentions in the query
        # Exact column matches get highest priority; partial / token matches of ID columns get penalized.
        scored_cols: list[tuple[int, str]] = []
        for col_lower, original_col in col_map.items():
            score = 0
            # Exact full column match: e.g., 'order_status'
            if re.search(rf"\b{re.escape(col_lower)}\b", q_lower):
                score += 100
            # Spaced match: e.g., 'order status' for 'order_status'
            col_spaced = col_lower.replace("_", " ")
            if re.search(rf"\b{re.escape(col_spaced)}\b", q_lower):
                score += 95

            # If no full match, check meaningful sub-tokens (avoid generic English words like 'order', 'date', 'count')
            if score == 0:
                _COMMON_WORDS = ("order", "orders", "data", "dataset", "value", "values", "total", "average", "count", "counts", "date", "time", "id", "item", "items", "group")
                tokens = [t for t in col_lower.split("_") if len(t) >= 4 and t not in _COMMON_WORDS]
                for tok in tokens:
                    if re.search(rf"\b{re.escape(tok)}\b", q_lower):
                        score += 40
                        break

            # Boost non-ID columns over ID columns
            if original_col in id_cols:
                # Only keep ID column if explicitly written with '_id' or ' id'
                if score < 95:
                    score = 0
                else:
                    score -= 20

            if score > 0:
                scored_cols.append((score, original_col))

        # Sort candidate columns by match score descending
        scored_cols.sort(key=lambda x: x[0], reverse=True)
        candidate_cols = [c for _, c in scored_cols]

        # 3. Assign X and Y roles based on chart requirements
        is_scatter = any(w in q_lower for w in ("scatter", "relationship between", "correlation between", "versus", "vs "))
        is_count_request = any(w in q_lower for w in ("count", "counts", "number of", "volume", "distribution of", "frequency", "how many"))

        if is_scatter:
            # Scatter plots require two numeric columns
            mentioned_numeric = [c for c in candidate_cols if c in numeric_cols]
            if len(mentioned_numeric) >= 2:
                return resolved_x or mentioned_numeric[0], resolved_y or mentioned_numeric[1]
            elif len(numeric_cols) >= 2:
                # Default to the first two numeric columns in the dataset
                return resolved_x or numeric_cols[0], resolved_y or numeric_cols[1]

        if resolved_x:
            # Grouping column is identified (e.g. order_status, date, department)
            if not resolved_y:
                # Check if a specific numeric metric was mentioned
                numeric_candidates = [c for c in candidate_cols if c in numeric_cols and c != resolved_x]
                if numeric_candidates and not is_count_request:
                    resolved_y = numeric_candidates[0]
                elif is_count_request or not numeric_cols:
                    resolved_y = "count"
                else:
                    resolved_y = numeric_cols[0]
            return resolved_x, resolved_y

        # If resolved_x not found via syntactic grouping, resolve from scored candidates
        if candidate_cols:
            cand_cat = [c for c in candidate_cols if c in categorical_cols or c in dt_cols]
            cand_num = [c for c in candidate_cols if c in numeric_cols]

            if cand_cat and cand_num:
                resolved_x = resolved_x or cand_cat[0]
                resolved_y = resolved_y or cand_num[0]
            elif cand_cat:
                resolved_x = resolved_x or cand_cat[0]
                resolved_y = resolved_y or (numeric_cols[0] if (numeric_cols and not is_count_request) else "count")
            elif cand_num:
                if len(cand_num) >= 2 and is_scatter:
                    resolved_x, resolved_y = cand_num[0], cand_num[1]
                else:
                    # Pick best companion categorical or datetime column
                    if dt_cols:
                        resolved_x = dt_cols[0]
                    elif categorical_cols:
                        resolved_x = categorical_cols[0]
                    else:
                        resolved_x = cols[0]
                    resolved_y = cand_num[0]

        # 4. Fallback defaults from dataset schema
        if not resolved_x:
            if dt_cols:
                resolved_x = dt_cols[0]
            elif categorical_cols:
                resolved_x = categorical_cols[0]
            elif cols:
                resolved_x = cols[0]

        if not resolved_y:
            if numeric_cols:
                candidates = [c for c in numeric_cols if c != resolved_x]
                resolved_y = candidates[0] if candidates else numeric_cols[0]
            else:
                resolved_y = "count"

        return resolved_x, resolved_y

    def _find_missing_explicit_fields(
        self,
        df: pd.DataFrame,
        q_lower: str,
        explicit_x: str | None,
        explicit_y: str | None,
    ) -> list[str]:
        """Identify fields explicitly requested by the user that do not exist in the dataset."""
        cols_lower = {c.lower() for c in df.columns}
        missing = []
        if explicit_x and explicit_x.lower() not in cols_lower:
            missing.append(explicit_x)
        if explicit_y and explicit_y.lower() not in cols_lower and explicit_y.lower() != "count":
            missing.append(explicit_y)

        # Check for phrases like "by <column>", "relationship between <col1> and <col2>", "plot <col>"
        m_by = re.search(r"\b(?:by|per|across|over|versus|vs\.?)\s+([a-zA-Z0-9_]+)\b", q_lower)
        if m_by:
            target = m_by.group(1).lower()
            if target not in cols_lower and target not in ("time", "month", "monthly", "year", "yearly", "date", "category", "product", "all", "chart", "group", "status", "day", "daily"):
                missing.append(target)

        return sorted(list(set(missing)))

    def _detect_datetime_columns(self, df: pd.DataFrame) -> list[str]:
        """Detect datetime or temporal string columns in the DataFrame."""
        dt_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                dt_cols.append(col)
                continue
            if any(term in col.lower() for term in ("date", "time", "timestamp", "year", "month", "day")):
                # Verify that it can actually parse as datetime
                if len(df) > 0:
                    sample = df[col].dropna().head(5)
                    if not sample.empty:
                        try:
                            pd.to_datetime(sample, format="mixed")
                            dt_cols.append(col)
                            continue
                        except Exception:
                            pass
                dt_cols.append(col)
                continue
            if df[col].dtype == object and len(df) > 0:
                sample = df[col].dropna().head(5)
                if not sample.empty:
                    try:
                        pd.to_datetime(sample, format="mixed")
                        dt_cols.append(col)
                    except Exception:
                        pass
        return dt_cols

    def _resolve_chart_type(
        self,
        df: pd.DataFrame,
        q_lower: str,
        explicit_type: str | None,
        x_col: str | None,
        y_col: str | None,
    ) -> ChartType:
        """Infer or confirm appropriate chart type from user intent and column data types."""
        if explicit_type and explicit_type.lower() in ("bar", "line", "pie", "donut", "scatter", "table", "kpi"):
            return explicit_type.lower()  # type: ignore

        if any(w in q_lower for w in ("scatter", "relationship between", "correlation between", "versus", "vs ")):
            return "scatter"
        if any(w in q_lower for w in ("pie", "donut", "share of", "proportion", "breakdown of")):
            return "pie"
        if any(w in q_lower for w in ("trend", "over time", "monthly", "yearly", "daily", "timeline", "line chart", "time series")):
            return "line"
        if any(w in q_lower for w in ("table", "preview", "raw data", "tabular summary")):
            return "table"
        if any(w in q_lower for w in ("kpi", "metrics", "summary card", "total count", "overall metrics")):
            return "kpi"

        # Infer based on schema data types
        dt_cols = self._detect_datetime_columns(df)
        if x_col and x_col in dt_cols:
            return "line"

        if x_col and y_col and pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
            return "scatter"

        return "bar"

    def _build_single_visualization(
        self,
        df: pd.DataFrame,
        chart_type: ChartType,
        x_col: str | None,
        y_col: str | None,
        query: str,
        aggregation: str | None = None,
        limit: int | None = None,
    ) -> VisualizationPayload:
        """Execute calculations and construct a fully grounded visualization specification."""
        clean_df = df.copy()
        q_lower = query.lower()

        # KPI chart
        if chart_type == "kpi":
            kpis = []
            kpis.append(KpiCard(label="Total Records", value=len(clean_df), subtext="Total rows in dataset"))
            numeric_cols = [c for c in clean_df.columns if pd.api.types.is_numeric_dtype(clean_df[c])]
            for col in numeric_cols[:3]:
                col_sum = clean_df[col].dropna().sum()
                col_avg = clean_df[col].dropna().mean()
                val_str = f"{col_sum:,.2f}" if isinstance(col_sum, float) else f"{col_sum:,}"
                avg_str = f"Avg: {col_avg:.2f}"
                kpis.append(KpiCard(label=f"Total {col}", value=val_str, subtext=avg_str))
            return VisualizationPayload(
                chart_type="kpi",
                title="Dataset Key Performance Indicators",
                description="Summary metrics computed directly from the tabular dataset.",
                kpis=kpis,
                columns=list(clean_df.columns),
            )

        # Table view
        if chart_type == "table":
            max_rows = limit or 15
            preview = clean_df.head(max_rows).fillna("").to_dict(orient="records")
            return VisualizationPayload(
                chart_type="table",
                title="Dataset Summary Table",
                description=f"Showing top {len(preview)} records.",
                data=preview,
                columns=list(clean_df.columns),
            )

        # Require at least one field for 2D charts
        if not x_col and not y_col:
            x_col = clean_df.columns[0]

        # Scatter plot
        if chart_type == "scatter":
            numeric_cols = [c for c in clean_df.columns if pd.api.types.is_numeric_dtype(clean_df[c])]
            if not x_col or not y_col or x_col not in numeric_cols or y_col not in numeric_cols:
                if len(numeric_cols) >= 2:
                    x_col, y_col = numeric_cols[0], numeric_cols[1]
                else:
                    return VisualizationPayload(
                        chart_type="table",
                        title="Scatter Plot Error",
                        error="Scatter plot requires at least two numeric fields in the dataset.",
                        columns=list(clean_df.columns),
                    )
            scatter_df = clean_df[[x_col, y_col]].dropna().head(limit or 100)
            data_points = []
            for _, row in scatter_df.iterrows():
                vx = float(row[x_col])
                vy = float(row[y_col])
                lbl_x = int(vx) if vx.is_integer() else round(vx, 2)
                lbl_y = int(vy) if vy.is_integer() else round(vy, 2)
                data_points.append({"x": vx, "y": vy, "label": f"({lbl_x}, {lbl_y})"})
            return VisualizationPayload(
                chart_type="scatter",
                title=f"Relationship between {x_col} and {y_col}",
                description=f"Scatter distribution of {y_col} vs {x_col} ({len(data_points)} data points).",
                x_field=x_col,
                y_field=y_col,
                data=data_points,
                columns=[x_col, y_col],
            )

        # Line Chart with Datetime Aggregation
        dt_cols = self._detect_datetime_columns(clean_df)
        if chart_type == "line" and x_col and x_col in dt_cols:
            try:
                clean_df["_dt_parsed"] = pd.to_datetime(clean_df[x_col], errors="coerce")
                clean_df = clean_df.dropna(subset=["_dt_parsed"])
                if any(w in q_lower for w in ("month", "monthly")):
                    clean_df["_time_period"] = clean_df["_dt_parsed"].dt.to_period("M").astype(str)
                    period_label = " (Monthly)"
                elif any(w in q_lower for w in ("year", "yearly")):
                    clean_df["_time_period"] = clean_df["_dt_parsed"].dt.to_period("Y").astype(str)
                    period_label = " (Yearly)"
                elif any(w in q_lower for w in ("day", "daily")):
                    clean_df["_time_period"] = clean_df["_dt_parsed"].dt.to_period("D").astype(str)
                    period_label = " (Daily)"
                else:
                    clean_df["_time_period"] = clean_df[x_col].astype(str)
                    period_label = ""

                if y_col and y_col != "count" and y_col in clean_df.columns and pd.api.types.is_numeric_dtype(clean_df[y_col]):
                    agg_fn = "mean" if any(w in q_lower for w in ("average", "avg", "mean")) else (aggregation or "sum")
                    res_series = clean_df.groupby("_time_period", sort=False)[y_col].agg(agg_fn)
                    metric_name = "Average" if agg_fn == "mean" else "Total"
                    title = f"{metric_name} {y_col} over {x_col}{period_label}"
                else:
                    res_series = clean_df["_time_period"].value_counts(sort=False)
                    y_col = "count"
                    title = f"Order Volume over {x_col}{period_label}" if "order" in q_lower else f"Volume over {x_col}{period_label}"

                data_points = []
                for p_val, v_val in res_series.items():
                    num_val = float(v_val) if isinstance(v_val, (int, float, np.number)) else 0.0
                    val_out = int(num_val) if num_val.is_integer() else round(num_val, 2)
                    data_points.append({
                        str(x_col): str(p_val),
                        str(y_col): val_out,
                        "label": str(p_val),
                        "value": val_out,
                    })

                return VisualizationPayload(
                    chart_type="line",
                    title=title,
                    description=f"Aggregation over temporal field {x_col}.",
                    x_field=x_col,
                    y_field=y_col,
                    data=data_points,
                    columns=list(df.columns),
                )
            except Exception as dt_err:
                logger.warning("Datetime parsing fallback for %s: %s", x_col, dt_err)

        # Bar, Pie / Donut, or standard Line chart
        is_count = (not y_col) or (y_col == "count") or (y_col == x_col) or (y_col in clean_df.columns and not pd.api.types.is_numeric_dtype(clean_df[y_col]))

        if is_count:
            # Frequency distribution of x_col
            res_series = clean_df[x_col].dropna().value_counts()
            agg_func = "count"
            effective_y = "count"
        else:
            agg_func = "mean" if any(w in q_lower for w in ("average", "avg", "mean")) else (aggregation or "sum")
            clean_df[y_col] = pd.to_numeric(clean_df[y_col], errors="coerce")
            grouped = clean_df.dropna(subset=[x_col, y_col]).groupby(x_col)[y_col]
            if agg_func == "mean":
                res_series = grouped.mean()
            else:
                res_series = grouped.sum()
            effective_y = y_col

        max_items = limit or (10 if chart_type == "bar" else (8 if chart_type in ("pie", "donut") else 30))
        if chart_type != "line":
            res_series = res_series.sort_values(ascending=False).head(max_items)
        else:
            res_series = res_series.head(max_items)

        data_points = []
        for x_val, y_val in res_series.items():
            num_val = float(y_val) if isinstance(y_val, (int, float, np.number)) else 0.0
            val_out = int(num_val) if num_val.is_integer() else round(num_val, 2)
            data_points.append({
                str(x_col): str(x_val),
                str(effective_y): val_out,
                "label": str(x_val),
                "value": val_out,
            })

        if agg_func == "count":
            title = f"Order Counts by {x_col}" if "order" in q_lower else f"Distribution of {x_col}"
        elif agg_func == "mean":
            title = f"Average {effective_y} by {x_col}"
        else:
            title = f"Total {effective_y} by {x_col}"

        return VisualizationPayload(
            chart_type=chart_type,
            title=title,
            description=f"{agg_func.capitalize()} calculation grouped by {x_col}.",
            x_field=x_col,
            y_field=effective_y,
            data=data_points,
            columns=list(clean_df.columns),
        )

    def _generate_multi_visualizations(
        self,
        df: pd.DataFrame,
        num_charts: int = 3,
        query: str = "",
    ) -> list[dict[str, Any]]:
        """Generate multiple diverse, grounded visualization specifications for comprehensive analysis.

        Builds a sequential candidate pool of distinct, schema-validated visualization candidates,
        deduplicating equivalent charts and prioritizing graphical charts when explicitly requested.
        """
        q_lower = query.lower()
        wants_pure_charts = bool(re.search(r"\b(?:charts?|plots?|graphs?)\b", q_lower)) and not bool(re.search(r"\b(?:visualizations?|summary|kpis?|metrics?)\b", q_lower))

        dt_cols = self._detect_datetime_columns(df)
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in dt_cols]
        # Meaningful categorical columns (exclude ID columns)
        categorical_cols = [c for c in df.columns if c not in numeric_cols and c not in dt_cols and not self._is_id_column(c, df)]

        # 1. KPI summary card
        kpi_cand = self._build_single_visualization(df, "kpi", None, None, "kpi summary")

        # 2. Primary Categorical Bar chart
        bar_primary = None
        if categorical_cols and numeric_cols:
            bar_primary = self._build_single_visualization(
                df, "bar", categorical_cols[0], numeric_cols[0], f"Total {numeric_cols[0]} by {categorical_cols[0]}"
            )
        elif categorical_cols:
            bar_primary = self._build_single_visualization(
                df, "bar", categorical_cols[0], "count", f"Distribution of {categorical_cols[0]}"
            )

        # 3. Temporal Line chart (if temporal column exists)
        line_cand = None
        if dt_cols:
            line_cand = self._build_single_visualization(
                df, "line", dt_cols[0], numeric_cols[0] if numeric_cols else "count", f"Trend over {dt_cols[0]}"
            )

        # 4. Numeric Scatter chart (if 2+ numeric columns exist)
        scatter_cand = None
        if len(numeric_cols) >= 2:
            scatter_cand = self._build_single_visualization(
                df, "scatter", numeric_cols[0], numeric_cols[1], f"Relationship between {numeric_cols[0]} and {numeric_cols[1]}"
            )

        # 5. Categorical Pie / Donut chart (if categorical column with <= 8 categories exists)
        pie_cand = None
        pie_col = None
        if len(categorical_cols) > 1 and df[categorical_cols[1]].nunique() <= 8:
            pie_col = categorical_cols[1]
        elif categorical_cols and df[categorical_cols[0]].nunique() <= 8:
            pie_col = categorical_cols[0]
        if pie_col:
            pie_cand = self._build_single_visualization(
                df, "pie", pie_col, "count", f"Breakdown of {pie_col}"
            )

        # 6. Secondary metric or dimension Bar chart (if additional numeric or categorical columns exist)
        secondary_bar = None
        if len(numeric_cols) >= 2 and categorical_cols:
            secondary_bar = self._build_single_visualization(
                df, "bar", categorical_cols[0], numeric_cols[1], f"Total {numeric_cols[1]} by {categorical_cols[0]}"
            )
        elif len(categorical_cols) >= 2 and numeric_cols:
            secondary_bar = self._build_single_visualization(
                df, "bar", categorical_cols[1], numeric_cols[0], f"Total {numeric_cols[0]} by {categorical_cols[1]}"
            )

        # 7. Table preview
        table_cand = self._build_single_visualization(df, "table", None, None, "Data Table Preview")

        # Assemble candidate pool based on whether pure graphical charts vs generic visualizations are requested
        if wants_pure_charts:
            ordered_candidates = [
                bar_primary,
                line_cand,
                scatter_cand,
                pie_cand,
                secondary_bar,
                kpi_cand,
                table_cand,
            ]
        else:
            ordered_candidates = [
                kpi_cand,
                bar_primary,
                line_cand,
                scatter_cand,
                pie_cand,
                secondary_bar,
                table_cand,
            ]

        # Deduplicate and validate against dataset
        specs: list[VisualizationPayload] = []
        seen_signatures: set[tuple[str, str | None, str | None]] = set()

        for cand in ordered_candidates:
            if cand is None or cand.error:
                continue
            sig = (cand.chart_type, cand.x_field, cand.y_field)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            specs.append(cand)
            if len(specs) >= num_charts:
                break

        return [s.model_dump() for s in specs]

