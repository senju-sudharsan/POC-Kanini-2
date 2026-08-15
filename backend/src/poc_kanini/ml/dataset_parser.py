"""Robust parser for inline dataset definitions in user chat messages."""

import ast
import csv
import io
import json
import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def _parse_scalar(val: str) -> Any:
    """Parse a string representation of a scalar value into typed Python value."""
    s = val.strip().strip("'\"")
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("none", "null", "nan", "na", ""):
        return None
    try:
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def parse_inline_dataset(text: str) -> list[dict[str, Any]] | str | None:
    """Extract a structured tabular dataset (list of record dicts or CSV string) from free text.
    
    Supports:
    1. JSON or Python literal list of dicts (e.g. `[{"x": 1, "y": 0}, ...]` or `[{'x': 1, 'y': 0}, ...]`)
    2. Parenthesized tuple records with headers:
       e.g. `dataset: feature1, feature2, churn. Use these records: (1,2,0), (2,3,0), (8,9,1), (9,10,0)`
    3. Multiline CSV blocks embedded in text
    4. Markdown table blocks
    """
    if not text or not text.strip():
        return None

    cleaned_text = text.strip()

    # 1. Check for JSON or Python literal list of dicts
    json_match = re.search(r"(\[\s*\{[\s\S]*\}\s*\])", cleaned_text)
    if json_match:
        raw_json = json_match.group(1)
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                return parsed
        except Exception:
            try:
                parsed = ast.literal_eval(raw_json)
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    return parsed
            except Exception:
                pass

    # 2. Check for parenthesized tuple records: e.g. (1,2,0), (2,3,0), ...
    tuple_matches = list(re.finditer(r"\(\s*([^\(\)]+)\s*\)", cleaned_text))
    if len(tuple_matches) >= 2:
        parsed_rows = []
        for tm in tuple_matches:
            raw_inside = tm.group(1).strip()
            # Split items by comma
            items = [_parse_scalar(item) for item in raw_inside.split(",")]
            if len(items) >= 2:
                parsed_rows.append(items)

        if len(parsed_rows) >= 2:
            row_len = len(parsed_rows[0])
            # Ensure rows are consistent in length
            consistent_rows = [r for r in parsed_rows if len(r) == row_len]
            if len(consistent_rows) >= 2:
                # Find column names from text before the first tuple
                first_tuple_pos = tuple_matches[0].start()
                pre_text = cleaned_text[:first_tuple_pos].strip()

                cols = []
                # Look for header keywords like "dataset: col1, col2, col3" or "features: ..."
                header_match = re.search(
                    r"(?:using\s+(?:this\s+)?dataset|dataset|columns?|features?|headers?|fields?)\s*:\s*([^.\n;]+)",
                    pre_text,
                    re.IGNORECASE,
                )
                if header_match:
                    raw_cols = header_match.group(1).strip()
                    cols = [c.strip().strip("'\"") for c in re.split(r"[,;]\s*", raw_cols) if c.strip()]

                if not cols or len(cols) != row_len:
                    # Look for comma-separated identifiers near the end of pre_text
                    # e.g. "... feature1, feature2, churn. Use these records:"
                    words_match = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b", pre_text)
                    # Exclude common instruction words
                    stop_words = {
                        "train", "a", "classification", "regression", "model", "using", "this",
                        "dataset", "use", "these", "records", "record", "data", "report", "the",
                        "metrics", "and", "feature", "importance", "please", "with", "target", "column",
                    }
                    candidate_cols = [w for w in words_match if w.lower() not in stop_words]
                    if len(candidate_cols) >= row_len:
                        cols = candidate_cols[-row_len:]

                if not cols or len(cols) != row_len:
                    # Fallback column naming
                    cols = [f"feature{i+1}" for i in range(row_len - 1)] + ["target"]

                return [dict(zip(cols, row)) for row in consistent_rows]

    # 3. Check for Markdown table: | col1 | col2 |
    md_lines = [line.strip() for line in cleaned_text.splitlines() if line.strip().startswith("|") and line.strip().endswith("|")]
    if len(md_lines) >= 3:
        header_cells = [c.strip() for c in md_lines[0].strip("|").split("|")]
        data_rows = []
        for line in md_lines[1:]:
            if re.match(r"^\|?\s*[-:\s|]+\s*\|?$", line):
                continue
            cells = [_parse_scalar(c) for c in line.strip("|").split("|")]
            if len(cells) == len(header_cells):
                data_rows.append(dict(zip(header_cells, cells)))
        if len(data_rows) >= 2:
            return data_rows

    # 4. Check for Multiline CSV text
    csv_lines = [line.strip() for line in cleaned_text.splitlines() if line.strip() and "," in line]
    if len(csv_lines) >= 3:
        try:
            reader = csv.reader(csv_lines)
            rows = list(reader)
            if len(rows) >= 3 and len(rows[0]) >= 2:
                header = [c.strip() for c in rows[0]]
                if all(re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", h) for h in header):
                    valid_rows = [
                        dict(zip(header, [_parse_scalar(cell) for cell in r]))
                        for r in rows[1:]
                        if len(r) == len(header)
                    ]
                    if len(valid_rows) >= 2:
                        return valid_rows
        except Exception:
            pass

    return None


def parse_data_to_dataframe(data: list[dict[str, Any]] | str) -> pd.DataFrame:
    """Parse list of dicts, CSV string, or inline prompt string into a pandas DataFrame."""
    if isinstance(data, list):
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)
    elif isinstance(data, str):
        # First check if it's already a clean CSV string
        try:
            df = pd.read_csv(io.StringIO(data))
            if not df.empty and len(df.columns) >= 2 and len(df) >= 1:
                return df
        except Exception:
            pass

        # Try extracting inline dataset from free text
        extracted = parse_inline_dataset(data)
        if isinstance(extracted, list):
            return pd.DataFrame(extracted)
        elif isinstance(extracted, str):
            return pd.read_csv(io.StringIO(extracted))

        # Fallback: re-try pd.read_csv
        return pd.read_csv(io.StringIO(data))
    else:
        raise ValueError("Unsupported data input type. Must be a list of records or a CSV string.")


def extract_prediction_features(text: str, expected_features: list[str] | None = None) -> list[dict[str, Any]] | None:
    """Extract feature dictionary or record list from a prediction query string.
    
    Supports:
    1. Key-value pairs: e.g. `feature1=8 and feature2=9` or `feature1: 8, feature2: 9`
    2. JSON / Python literal dict or list: `[{"feature1": 8, "feature2": 9}]` or `{"feature1": 8, "feature2": 9}`
    3. Positional values in tuple or list: `(8, 9)` or `[8, 9]`
    """
    if not text or not text.strip():
        return None

    cleaned_text = text.strip()

    # 1. JSON / Python literal dict or list of dicts
    json_match = re.search(r"(\[\s*\{[\s\S]*\}\s*\]|\{[\s\S]*\})", cleaned_text)
    if json_match:
        raw_json = json_match.group(1)
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                return parsed
        except Exception:
            try:
                parsed = ast.literal_eval(raw_json)
                if isinstance(parsed, dict):
                    return [parsed]
                if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict):
                    return parsed
            except Exception:
                pass

    # 2. Key-value pairs: e.g. feature1=8 and feature2=9 or feature1: 8, feature2: 9
    kv_matches = re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*[:=]\s*([0-9\.\-\+eE]+|true|false|'[^']*'|\"[^\"]*\")", cleaned_text)
    if kv_matches:
        record = {}
        for k, v in kv_matches:
            record[k] = _parse_scalar(v)
        if record:
            return [record]

    # 3. Tuple of numbers: e.g. (8, 9) or [8, 9] (avoid matching single numbers in text)
    tuple_match = re.search(r"[\(\[]\s*([0-9\.\-\+eE\s,]+)\s*[\)\]]", cleaned_text)
    if tuple_match:
        numbers = [_parse_scalar(n) for n in tuple_match.group(1).split(",") if n.strip()]
        if len(numbers) >= 1:
            if expected_features and len(expected_features) == len(numbers):
                return [dict(zip(expected_features, numbers))]
            else:
                cols = [f"feature{i+1}" for i in range(len(numbers))]
                return [dict(zip(cols, numbers))]

    return None
