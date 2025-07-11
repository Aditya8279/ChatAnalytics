from django.shortcuts import render
from .llm_pipeline import anomalies_agent, data_identifier_agent, table_schema_agent, generate_greet_output, classification_agent, break_into_subquestions, generate_python_code, generate_plot_code, generate_summary, generate_final_summary,generate_title, generate_description
import pandas as pd
from pandas.api.types import is_period_dtype
import numpy as np
import requests
import time
import json
import openai

import plotly.express as px

import difflib
import re

import logging

import logging

import pandas as pd
from pandas import Timestamp

import io
from PIL import Image
import matplotlib.pyplot as plt
import base64

from fpdf import FPDF
import tempfile

from datetime import datetime
import os
# from django.http import HttpResponse

from sklearn.ensemble import IsolationForest

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.utils.timezone import now

def extract_metadata(df):
    sample = df.head(3).copy()
    
    # Convert potentially non-serializable types to strings
    sample = sample.applymap(lambda x: str(x) if isinstance(x, (pd.Timestamp, pd.Timedelta, np.generic)) else x)
    
    # Convert sample DataFrame to list of dicts for JSON serialization
    sample_dict = sample.to_dict(orient="records")

    info = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "samples": sample_dict  # <- use the dict, not the DataFrame
    }

    return json.dumps(info, indent=2)

def fuzzy_fix_code(python_code: str, df: pd.DataFrame) -> str:
    """Replace hardcoded string literals in code with most similar actual values from df."""
    string_literals = re.findall(r"'(.*?)'", python_code)
    for literal in string_literals:
        for col in df.columns:
            if df[col].dtype == "object":
                unique_vals = df[col].dropna().unique().astype(str)
                matches = difflib.get_close_matches(literal, unique_vals, n=1, cutoff=0.6)
                if matches:
                    corrected = matches[0]
                    if corrected != literal:
                        python_code = python_code.replace(f"'{literal}'", f"'{corrected}'")
    return python_code

def extract_json_from_response(response):
    print(f"\n[DEBUG] Input type = {type(response)}")

    # If already parsed as dict, return it
    if isinstance(response, dict):
        print("[DEBUG] Already a dict.")
        return response

    if isinstance(response, str):
        # If wrapped in triple backticks, extract contents
        code_block_pattern = re.compile(r"```(?:json|python)?\s*([\s\S]+?)\s*```", re.MULTILINE)
        match = code_block_pattern.search(response)
        if match:
            response = match.group(1).strip()
        
        return response

    raise TypeError("❌ Model response must be a string or dict.")


# Precompile regex pattern once
special_char_pattern = re.compile(r"[^A-Za-z0-9.\s\-]")

def remove_special_chars_series(series):
    return series.astype(str).str.replace(special_char_pattern, "", regex=True)

def clean_special_chars(df):
    obj_cols = df.select_dtypes(include='object').columns
    df[obj_cols] = df[obj_cols].apply(remove_special_chars_series)
    return df


def convert_string_numerics_fast(df):
    obj_cols = df.select_dtypes(include='object').columns
    for col in obj_cols:
        s = df[col].astype(str).str.strip()
        # Try converting only if at least one value looks numeric
        if s.str.match(r"^[\d\.\-]+$").sum() > 0:
            converted = pd.to_numeric(s, errors='coerce')
            if converted.notna().sum() > 0:
                df[col] = converted
    return df

def inject_plot_saving(code: str) -> str:
    if "plt.show()" in code:
        code = code.replace("plt.show()", "plt.savefig('plot.png'); plt.close()")
    elif "plt.plot" in code or "sns." in code:
        code += "\nplt.savefig('plot.png')\nplt.close()"
    return code

def sub_summaries_to_text(summaries):
    return "\n\n".join(
        f"Insight {i+1}: {summary}" for i, summary in enumerate(summaries)
    )

def format_df_summary_table(df: pd.DataFrame) -> str:
    lines = ["Column_Name\tSample Value1\tSample Value2\tSample Value3"]
    for col in df.columns:
        samples = df[col].dropna().astype(str).tolist()[:3]
        samples += [""] * (3 - len(samples))  # pad if < 3 values
        line = f"{col}\t{samples[0]}\t{samples[1]}\t{samples[2]}"
        lines.append(line)
    return "\\n".join(lines)

def format_df_summary_table_as_markdown(df: pd.DataFrame) -> str:
    lines = ["| Column Name | Sample Value1 | Sample Value2 | Sample Value3 |",
             "|-------------|----------------|----------------|----------------|"]
    for col in df.columns:
        samples = df[col].dropna().astype(str).tolist()[:3]
        samples += [""] * (3 - len(samples))
        lines.append(f"| {col} | {samples[0]} | {samples[1]} | {samples[2]} |")
    return "\n".join(lines)



def fix_llm_code(raw_code: str) -> str:

    raw_code = raw_code.replace('\\n', '\n')
    #Prepend import if missing
    if "import pandas" not in raw_code:
        raw_code = "import pandas as pd\n" + raw_code

    lines = raw_code.strip().split('\n')

    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        #Remove plotting logic lines
        if re.search(r"\.plot\s*\(", line):
            # Try to convert to proper assignment if possible
            if '=' in line:
                left, right = line.split('=', 1)
                # cleaned_right = re.sub(r"\.unstack\(\)(\.plot\([^)]*\))?", "", right.strip())
                cleaned_right = re.sub(r"\.unstack\(\)\.plot\(.*\)", "", right.strip())
                cleaned_right = re.sub(r"\.plot\(.*\)", "", cleaned_right)
                if cleaned_right:
                    new_lines.append(f"result = {cleaned_right}")
                else:
                    new_lines.append("result = None")
            i += 1
            continue

        new_lines.append(line)
        i += 1

    #Fallback assignment to `result` if still missing
    if not any(re.match(r"^\s*result\s*=", line) for line in new_lines):
        for i in reversed(range(len(new_lines))):
            line = new_lines[i].strip()
            if (
                line and
                not line.startswith("#") and
                not re.match(r"^(if|else|elif|for|while|def|class)\b", line) and
                not line.endswith(":") and
                "result" not in line
            ):
                match = re.match(r"^(\w+)\s*=", line)
                if match:
                    var = match.group(1)
                    new_lines.insert(i + 1, f"result = {var}")
                    break
        else:
            new_lines.append("result = None")

    return "\n".join(new_lines)

def replace_dataframe_var(code: str) -> str:
    """
    Replaces 'data' with 'result' in the line where pd.DataFrame(data) is used.
    Only changes the argument inside pd.DataFrame(), not variable names elsewhere.
    """
    # Regex to match: df = pd.DataFrame(data)
    pattern = r"(df\s*=\s*pd\.DataFrame\()\s*data\s*(\))"
    replaced_code = re.sub(pattern, r"\1result\2", code)
    return replaced_code


# Configure logging
logging.basicConfig(
    filename="model_outputs.log",  # log file name
    level=logging.INFO,            # log level (can use DEBUG for more granularity)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def inject_plot_formatting(code: str, height: int = 300) -> str:
    """
    Inject layout and HTML rendering into Plotly figure code.
    Also removes 'fig.show()' if present.
    """
    lines = code.strip().splitlines()
    modified_lines = []
    
    fig_started = False
    fig_complete = False
    fig_block = []
    is_line_plot = False

    for line in lines:
        stripped = line.strip()

        # Skip fig.show()
        if stripped.startswith("fig.show()"):
            continue

        # Detect start of fig assignment
        if not fig_started and stripped.startswith("fig = "):
            fig_started = True

        # Check if this is a line plot
        if "px.line" in line:
            is_line_plot = True

        if fig_started and not fig_complete:
            fig_block.append(line)
            if stripped.endswith(")") or stripped.endswith("),"):  # function call end
                fig_complete = True
                modified_lines.extend(fig_block)

                # Add dot markers for line plots
                if is_line_plot:
                    modified_lines.append('fig.update_traces(mode="lines+markers")')

                # Inject formatting
                modified_lines.append(
                    f"fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), autosize=True, height={height}, "
                    f"plot_bgcolor='white', paper_bgcolor='white', "
                    f"xaxis=dict(showgrid=False, showticklabels=False), "
                    f"yaxis=dict(showgrid=True, showticklabels=True, gridcolor='lightgrey', tickformat='.2~s'))"
                )
                modified_lines.append(
                    'plot_html = fig.to_html(full_html=False, include_plotlyjs=False, '
                    'config={"displayModeBar": False, "responsive": True})'
                )
        elif not fig_started:
            modified_lines.append(line)
        elif fig_complete:
            modified_lines.append(line)

    return "\n".join(modified_lines)


def safe_reset_index(df):
    
    df = pd.DataFrame(df)

    # Handle duplicated index names (e.g., multiple "date" levels)
    index_names = list(df.index.names)
    counts = {}
    new_index_names = []
    
    for name in index_names:
        if name in counts:
            counts[name] += 1
            new_name = f"{name}_{counts[name]}"
        elif name in df.columns:
            counts[name] = 1
            new_name = f"{name}_1"
        else:
            counts[name] = 0
            new_name = name
        new_index_names.append(new_name)
    
    df.index.names = new_index_names
    return df.reset_index()

def insert_user_question(sub_questions, user_question, analysis_info):
    question_type = analysis_info["question_type"]
    scope = analysis_info["scope"]

    if question_type != "Direct":
        return sub_questions  # Only direct questions should be inserted

    # Insert at the correct index based on scope
    if scope == "SingleValue":
        insert_index = min(4, len(sub_questions))  # Position 5 (0-indexed)
    elif scope == "MultipleValue":
        insert_index = min(6, len(sub_questions))  # Position 7
    else:
        return sub_questions  # Do not insert if scope is Unknown

    sub_questions.insert(insert_index, user_question)
    return sub_questions

def human_format(value):
    """
    Converts a numeric value to a human-readable string (e.g. 1,250,000 → 1.25M).
    Leaves non-numeric values unchanged.
    """
    try:
        num = float(value)
    except (ValueError, TypeError):
        return value  # Not a number

    abs_num = abs(num)

    if abs_num < 1_000:
        return f"{num:.2f}"
    elif abs_num < 1_000_000:
        return f"{num / 1_000:.2f}K"
    elif abs_num < 1_000_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif abs_num < 1_000_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    else:
        return f"{num / 1_000_000_000_000:.2f}T"
    

def detect_auto_anomalies(df, model_output, iqr_multiplier=4.5, iso_contamination=0.005):
    """
    Automatically detect anomalies in either time series or general tabular data.

    Parameters:
    - df: Input DataFrame
    - model_output: JSON dict from LLM containing keys:
        - is_timeseries: bool
        - datetime_column: str
        - target_columns: list of str
        - frequency: str (optional, for future tuning)

    Returns:
    - result_df: DataFrame with anomalies and 'Anomaly Reason' column
    - anomaly_flags: Boolean Series marking anomalies
    """
    
    is_timeseries = model_output.get("is_timeseries", False)
    target_cols = model_output.get("target_columns", [])

    # if not target_cols or (is_timeseries and not date_col):
    #     return {"error": "Model output missing required information."}, pd.Series([False] * len(df), index=df.index)

    df = df.copy()

    # ---------- TIME SERIES HANDLING ----------
    if is_timeseries:
        
        date_col = model_output.get("datetime_column")
        freq_map = {
            'daily': 7,        # 7-day window
            'hourly': 24,      # 24-hour window
            'weekly': 4,       # 4-week window (approx. 1 month)
            'monthly': 12,     # 12-month window (1 year)
            'quarterly': 4,    # 4 quarters (1 year)
            'yearly': 2        # 3-year window
        }

        detected_freq = model_output.get("frequency")
        # detected_freq = "Monthly"
        window = freq_map.get(detected_freq, 7)  # default to 7 if not found


        
        df[date_col] = pd.to_datetime(df[date_col].astype(str), errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.sort_values(by=date_col)

        # result_df = df.copy()
        # result_df["Anomaly Reason"] = ""

        anomaly_descriptions = []

        for col in target_cols:
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                continue
            temp_df = df[[date_col, col]].copy()
            temp_df['rolling_mean'] = temp_df[col].rolling(window=window, min_periods=1).mean()
            temp_df['rolling_std'] = temp_df[col].rolling(window=window, min_periods=1).std()
            temp_df['z_score'] = (temp_df[col] - temp_df['rolling_mean']) / temp_df['rolling_std']
            temp_df['anomaly'] = abs(temp_df['z_score']) > 3

            # result_df.loc[temp_df['anomaly'], "Anomaly Reason"] += f"TimeSeries Z-Score: {col}; "
            # Add both column name and value to the anomaly reason
            for idx in temp_df.index[temp_df['anomaly']]:
                date_val = temp_df.loc[idx, date_col]
                value = temp_df.loc[idx, col]
                # result_df.at[idx, "Anomaly Reason"] += (
                #     f"TimeSeries Z-Score (window={window}): {col}={value} on date column_name:'{date_col}'=column_value:'{date_val}'; "
                # )
                description = (
                    f"TimeSeries Z-Score (window={window}): {col}={value} on date column_name:'{date_col}'=column_value:'{date_val}';"
                )
                anomaly_descriptions.append(description)

        # flags = result_df["Anomaly Reason"] != ""
        # return result_df[flags], flags
        return anomaly_descriptions

    # ---------- NON-TIME SERIES HANDLING ----------
    else:
        numeric_df = df.select_dtypes(include='number')
        if numeric_df.empty:
            return ["No numeric columns for anomaly detection."]
    
        descriptions = []
    
        # --- IQR Detection ---
        def iqr_anomaly_flags(df, multiplier):
            flags = pd.Series([False] * len(df), index=df.index)
            reasons = pd.Series([""] * len(df), index=df.index)
            for col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - multiplier * IQR
                upper = Q3 + multiplier * IQR
                outliers = (df[col] < lower) | (df[col] > upper)
                flags |= outliers
                for idx in df.index[outliers]:
                    val = df.loc[idx, col]
                    reasons[idx] += f"{col}: {val}; "
            return flags, reasons
    
        # --- Isolation Forest Detection ---
        def isolation_forest_flags(df, contamination):
            clean_df = df.dropna()
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(clean_df)
            flags = pd.Series(False, index=df.index)
            reasons = pd.Series([""] * len(df), index=df.index)
            flags[clean_df.index] = preds == -1
    
            for idx in clean_df[flags[clean_df.index]].index:
                diffs = (clean_df.loc[idx] - clean_df.median()).abs()
                top_features = diffs.sort_values(ascending=False).head(3).index.tolist()
                reason_parts = [f"{col}: {df.loc[idx, col]}" for col in top_features]
                reasons[idx] = "IF: " + ", ".join(reason_parts)
            return flags, reasons
    
        # Run both detections
        flags_iqr, reasons_iqr = iqr_anomaly_flags(numeric_df, iqr_multiplier)
        flags_if, reasons_if = isolation_forest_flags(numeric_df, iso_contamination)
        combined_flags = flags_iqr | flags_if
    
        for idx in df.index[combined_flags]:
            parts = []
            if flags_iqr[idx]:
                parts.append("IQR: " + reasons_iqr[idx].strip())
            if flags_if[idx]:
                parts.append(reasons_if[idx].strip())
            description = " | ".join(parts)
            descriptions.append(description)
    
        return descriptions

@csrf_exempt
def upload_csv(request):
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')

        if csv_file:
            # Read and store CSV in session
            df = pd.read_csv(csv_file, dtype=str, low_memory=False)
            # Drop columns with > 50% missing values
            df = df.loc[:, df.isnull().mean() <= 0.5]
            df = df.dropna()  # drop rows with NaNs

            # # Step 2: Fill remaining missing values
            # for col in df.columns:
            #     # if pd.api.types.is_numeric_dtype(df[col]):
            #     #     df[col] = df[col].fillna(0)
            #     # else:
            #     df[col] = df[col].fillna(pd.NA)  # or None

            # Remove special characters from object columns only
            df = clean_special_chars(df)
            #Convert eligible object columns to numerics
            df = convert_string_numerics_fast(df)
            metadata = extract_metadata(df)

            request.session["csv_data"] = df.to_json()
            request.session["columns"] = list(df.columns)
            request.session["metadata"] = metadata

            meta_data = f"Metadata:\n{metadata}"
            logging.info(f"=== meta_data ===\n\n{meta_data}")
            table_schema_agent_response = table_schema_agent(meta_data)
            logging.info(f"=== table_schema_agent_response ===\n\n{table_schema_agent_response}")
            
            data_identifier_agent_response = data_identifier_agent(meta_data)
            data_identifier_agent_response = extract_json_from_response(data_identifier_agent_response)
            data_identifier_agent_response = json.loads(data_identifier_agent_response)

            anomalies_df = detect_auto_anomalies(df, data_identifier_agent_response)

            anomalies_agent_response = anomalies_agent(str(anomalies_df[-3:]))
            logging.info(f"=== anomalies_agent_response ===\n\n{anomalies_agent_response}")

            # Save to a file for download
            download_path = os.path.join(settings.MEDIA_ROOT, "cleaned_data.csv")
            df.to_csv(download_path, index=False)

            return JsonResponse({
                "message": f"✅ The {csv_file.name} dataset contains {df.shape[1]} columns and {df.shape[0]} rows after preprocessing.",
                "download_url": f"{settings.MEDIA_URL}cleaned_data.csv",
                "table-schema": table_schema_agent_response.strip(),
                "Anomalies":anomalies_agent_response
            })

        elif "csv_data" in request.session:
            # Step 2: Load existing CSV from session
            df = pd.read_json(request.session["csv_data"])
        else:
            df = None
            return JsonResponse({"error": "Invalid request"}, status=400)
        
def submit_schema(request):
    if request.method == 'POST':
        try:
            body = request.body.decode('utf-8')
            logging.info(f"Raw body received:\n{body}")

            if body:

                data = json.loads(body)
                updated_schema = data.get('updated_schema')
                request.session["table_schema"] = updated_schema

                logging.info(f"=== Received updated schema: ===\n\n{updated_schema}")
                return JsonResponse({'status': 'ok'})

        except json.JSONDecodeError:
            logging.error("Invalid JSON received.")
            return JsonResponse({'status': 'Error'})
        
def process_sub_question(i, sub_q, metadata, df, summaries, plot_paths, q_title, filter_result, description_list):

    logging.info(f"=== Enter loop with question number (Q{i+1}) ===\n\n{sub_q}")
    result = None
    last_model2_error = ""
    python_code = ""

    # Return code to filter dataframe based on sub_q
    prompt_2 = f"User Question: {sub_q}\n\nMetadata:\n{metadata}"
    for attempt in range(5):
        try:
            retry_prompt_2 = (
                f"{prompt_2}\n\nPrevious Output error (if any): {last_model2_error}"
                if last_model2_error else prompt_2
            )

            logging.info(f"=== Input | Data Filter Step for (Q{i+1}) ===\n\n")

            code_response = generate_python_code(retry_prompt_2)

            logging.info(f"=== Output | Data Filter Step for (Q{i+1}) ===\n\n{code_response}")

            # python_code = code_response
            code_response = extract_json_from_response(code_response)
            python_code = fix_llm_code(code_response)
                        
            local_vars = {"df": df.copy()}
            exec(python_code, {}, local_vars)
            result = local_vars.get("result")

            result = safe_reset_index(result)

            # filtered_data[i] = result
            if result is not None:
                break
        except Exception as e:
            last_model2_error = python_code+str(e)

    if isinstance(result, pd.DataFrame) and result.shape[0] > 1 and result.shape[1] > 1 or isinstance(result, pd.Series) and result.shape[0] > 1:

        for col in result.columns:
            if is_period_dtype(result[col]):
                result[col] = result[col].astype(str)

        result.loc[:, result.select_dtypes(include=['float', 'float64']).columns] = result.select_dtypes(include=['float', 'float64']).round(2)

        result_head = result.head(14)

        local_vars = {"df": result.copy()}
        summary_prompt = f"Dataset: \n{result_head.to_markdown(index=False)}"
        
    else:
        if isinstance(result, (int, float, np.integer, np.floating)):
            logging.info(f"=== result variable type (Q{i+1}, Attempt {attempt+1}) ===\n{type(result)}\n\n")
            result = round(result, 2)
            summary_prompt = f"User Query: {sub_q}\nOutput Value: \n{result}"
            description = generate_description(summary_prompt)
            description_list[i] = description
            result = human_format(result)
            filter_result[i] = result
            # filter_result.append(result)
        elif isinstance(result, Timestamp):
            logging.info(f"=== result variable type (Q{i+1}, Attempt {attempt+1}) ===\n{type(result)}\n\n")
            result = str(result)
            summary_prompt = f"User Query: {sub_q}\nOutput Value: \n{result}"
            description = generate_description(summary_prompt)
            description_list[i] = description
            filter_result[i] = result
        else:
            if not isinstance(result, (pd.DataFrame, pd.Series)):
                logging.info(f"=== result variable type (Q{i+1}, Attempt {attempt+1}) ===\n{type(result)}\n\n")
                summary_prompt = f"User Query: {sub_q}\nOutput Value: \n{result}"
                description = generate_description(summary_prompt)
                description_list[i] = description
                filter_result[i] = result
            else:
                summary_prompt = f"Dataset: \n{result.to_markdown(index=False)}"
                description = generate_description(summary_prompt)

    last_model3_error = ''
    viz_code_response = ''
    for attempt in range(3):
        try:

            retry_summary_prompt = (
                f"{summary_prompt}\n\nPrevious Output error (if any): {last_model3_error}"
                if last_model3_error else summary_prompt
            )
            # viz_prompt = f"""Dataset:\n{result.to_markdown(index=False)}"""
            logging.info(f"=== MODEL 3 Dashboard input (Q{i+1}, Attempt {attempt+1}) ===\n{retry_summary_prompt}\n\n")
            if isinstance(result, pd.DataFrame) and result.shape[0] > 1 and result.shape[1] > 1 or isinstance(result, pd.Series) and result.shape[0] > 1:
                viz_code_response = generate_plot_code(retry_summary_prompt)
                logging.info(f"=== MODEL 3 Dashboard Plot Code (Q{i+1}, Attempt {attempt+1}) ===\n{viz_code_response}\n\n")
                viz_code_response = extract_json_from_response(viz_code_response)
                viz_code_response = inject_plot_formatting(viz_code_response, height=280)
                logging.info(f"=== MODEL 3 Dashboard local_vars access (Q{i+1}, Attempt {attempt+1}) ===\n{local_vars}\n\n")
                exec(viz_code_response, {}, local_vars)

                # Extract HTML from variable (assume model always uses `plot_html`)
                plot_html = local_vars.get("plot_html", "")
                # plot_paths.append(plot_html)
                plot_paths[i] = plot_html
                # viz_code_response = query_llm(retry_summary_prompt, MODEL_3_SYSTEM_PROMPT)
            else:
                viz_code_response = generate_title(sub_q)
                logging.info(f"=== MODEL title Dashboard Output (Q{i+1}, Attempt {attempt+1}) ===\n{viz_code_response}\n\n")
                q_title[i] = viz_code_response
                # q_title.append(viz_code_response)
                logging.info(f"=== MODEL title Dashboard Output Saved")
                # viz_code_response = query_llm(retry_summary_prompt, MODEL_NO_DF_SYSTEM_PROMPT)
            
            # plot_paths.append(encoded_img)
            break
        except Exception as e:
            last_model3_error = viz_code_response+str(e)
            logging.warning(f"⚠️ MODEL 3 Attempt {attempt+1} failed: {e}")
            plt.close()

    summary_result = generate_summary(summary_prompt)

    # summaries.append({
    #     "question": sub_q,
    #     "summary": summary_result
    # })

    summaries[i] = {
        "question": sub_q,
        "summary": summary_result
    }

    # return summaries, plot_paths, q_title, filter_result, description_list

import concurrent.futures

def run_in_parallel(sub_questions, metadata, df):

    logging.info(f"=== Inside run_in_parallel function ===")
    
    num_qs = len(sub_questions)

    summaries = [None] * (num_qs)
    plot_paths = [None] * (num_qs)
    q_title = [None] * (num_qs)
    filter_result = [None] * (num_qs)
    description_list = [None] * (num_qs)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(
            lambda args: process_sub_question(args[0], args[1], metadata, df, summaries, plot_paths, q_title, filter_result, description_list),
            enumerate(sub_questions)
        )

    return summaries, plot_paths, q_title, filter_result, description_list



def dashboard_view(request):
    context = {}

    # if not request.session.session_key:
    #     request.session.create()  # force session to initialize
        
    # Detect fresh session or first use
    # if not request.session.get("initialized"):
    #     request.session["past_questions"] = []
    #     request.session["initialized"] = True

    # Reset past questions if session is new (first-time visit)
    if not request.session.get("visited_at1"):
        request.session["past_questions"] = []
        request.session["visited_at1"] = str(now())  # or use uuid if needed

    # No need to check again — it's already initialized above
    past_questions = request.session["past_questions"]

    top_insights=[]
    plot_images=[]
    final_summary=None
    combined_final_summary = None

    if request.method == 'POST':

        # Load from session if exists
        plot_images = request.session.get('plot_images', [])
        # logging.info(f"=== plot_image ===\n\n{plot_image}")
        top_insights = request.session.get('top_insights', [])
        # logging.info(f"=== top_insights ===\n\n{top_insights}")
        combined_final_summary = request.session.get('final_summary', None)
        # logging.info(f"=== combined_final_summary ===\n\n{combined_final_summary}")
        # ✅ Retrieve the existing list from session or initialize empty list
        past_questions = request.session.get("past_questions", [])


        user_question = request.POST.get('user_query', '')
        plot_question = request.POST.get('plot_question', '')
        action = request.POST.get('action')
        # logging.info(f"=== plot_question ===\n\n{plot_question}")

        metadata = request.session["metadata"]
        table_schema = request.session["table_schema"]
        

        logging.info(f"=== table_schema ===\n\n{table_schema}")
        

        # ✅ Add the new question if it's not empty
        if user_question.strip():
            # past_questions.append(user_question)
            if user_question.lower() not in [q.lower() for q in past_questions]:
                past_questions.insert(0, user_question)
                request.session["past_questions"] = past_questions
        elif plot_question.strip():
            user_question = plot_question
            sub_questions = [user_question]
        elif action == 'remove_plot':
            index = int(request.POST.get('plot_index', -1))
            if 0 <= index < len(plot_images):
                plot_images.pop(index)
                request.session['plot_images'] = plot_images

            context = {
                "past_questions": past_questions,  # list of previous questions
                "top_insights": top_insights,
                "plot_images": plot_images,
                "final_summary":combined_final_summary
            }
                
            return render(request, 'dashboard.html', context)
        
        elif action == "remove_tile":
            index = int(request.POST.get("tile_index", -1))
            if 0 <= index < len(top_insights):
                top_insights.pop(index)
                request.session['top_insights'] = top_insights

            context = {
                "past_questions": past_questions,  # list of previous questions
                "top_insights": top_insights,
                "plot_images": plot_images,
                "final_summary":combined_final_summary
            }
                
            return render(request, 'dashboard.html', context)

        # ✅ Load preprocessed DataFrame from session
        if "csv_data" in request.session:
            df = pd.read_json(request.session["csv_data"])
        else:
            return render(request, "dashboard.html", {
                "error": "❌ No CSV uploaded yet. Please upload a file first.",
                "past_questions": past_questions
            })
        
        # df = df.applymap(remove_special_chars)
        # df = convert_string_numerics(df)
    
        # logging.info(f"=== classification_agent output ===\n\n{analysis_info}")
        # # Parse the string into a dictionary
        # analysis_info = json.loads(analysis_info)

        if not plot_question.strip():
            user_query = f"User Question: {user_question}\n\nMetadata:\n{metadata}"


            logging.info(f"=== user_query ===\n\n{user_query}")
            sub_questions = break_into_subquestions(user_query)
            logging.info(f"=== Break down questions ===\n\n{sub_questions}")

        
        # updated_list = insert_user_question(sub_questions, user_question, analysis_info)
        # logging.info(f"=== Updated Break down questions list ===\n\n{updated_list}")

        # sub_questions = updated_list

        summaries, plot_paths, q_title, filter_result, description_list = run_in_parallel(sub_questions, metadata, df)
        plot_paths = [path for path in plot_paths if path is not None]
        if not plot_paths:
            plot_paths = [None]

        logging.info(f"=== MODEL summary Dashboard Input ===\n{summaries}\n\n")
        # combined_insights = "\n\n".join(summaries)

        combined_insights = "\n\n".join(
            f"Q: {item['question']}\nA: {item['summary']}"
            for item in summaries
            if item and item.get("summary") is not None
        )

        combined_insights = f"What are the most important insights or anomalies based on User's Question?\n\nUser Question:{user_question}\n\ncombined Q&A:\n{combined_insights}"
        final_summary = generate_final_summary(combined_insights)

        logging.info(f"=== MODEL final summary Dashboard Output ===\n{final_summary}\n\n")
        # Convert string to Python dict
        final_summary = extract_json_from_response(final_summary)
        
        if isinstance(final_summary, str):
            try:
                final_summary = json.loads(final_summary)
            except json.JSONDecodeError as e:
                print("JSON decoding failed:", e)
                final_summary = {}
        # final_summary = json.loads(final_summary)

        if plot_question.strip():
            if q_title[0] is not None and filter_result[0] is not None and description_list[0] is not None:
                top_insights.insert(0,{"label": q_title[0], "value": filter_result[0], "description": description_list[0]})

                # plot_paths = plot_images
                final_context = str(combined_final_summary) + str(final_summary)
                logging.info(f"=== MODEL final_context Dashboard Input ===\n{final_context}\n\n")
                final_summary = generate_final_summary(final_context)
                logging.info(f"=== MODEL final_context Dashboard Output ===\n{final_summary}\n\n")
                final_summary = extract_json_from_response(final_summary)

                if isinstance(final_summary, str):
                    try:
                        final_summary = json.loads(final_summary)
                    except json.JSONDecodeError as e:
                        print("JSON decoding failed:", e)
                        final_summary = {}
                        
            elif plot_paths[0] is not None:
                plot_images.insert(0, plot_paths[0])
                # plot_paths = plot_images
                final_context = str(combined_final_summary) + str(final_summary)
                logging.info(f"=== MODEL final_context Dashboard Input ===\n{final_context}\n\n")
                final_summary = generate_final_summary(final_context)
                logging.info(f"=== MODEL final_context Dashboard Output ===\n{final_summary}\n\n")
                final_summary = extract_json_from_response(final_summary)

                if isinstance(final_summary, str):
                    try:
                        final_summary = json.loads(final_summary)
                    except json.JSONDecodeError as e:
                        print("JSON decoding failed:", e)
                        final_summary = {}
                        
        else:
            plot_images = plot_paths

            # final_summary = json.loads(final_summary)


        # context = {
        #     'sub_questions': sub_questions,
        #     'filtered_tables': filtered_data,
        #     'plot_images': plot_paths,
        #     'user_question': user_query,
        #     'past_questions': [user_question],  # You can extend this with session/caching
        #     "loop_range": range(6),
        # }

        # context = {
        #     "results": zip(sub_questions, plot_paths, filtered_data, q_title),
        #     "past_questions": [user_question],
        # }


        if not plot_question.strip():
            top_insights = []
            for label, value, description in zip(q_title, filter_result, description_list):
                if label and value:  # Exclude if either is None, empty, or falsy
                    top_insights.append({"label": label, "value": value, "description": description})

    # Save to session
    logging.info(f"=== Save variables in session ===")
    request.session['plot_images'] = plot_images
    request.session['final_summary'] = final_summary
    request.session['top_insights'] = top_insights

    context = {
        "past_questions": past_questions,  # list of previous questions
        "top_insights": top_insights,
        "plot_images": plot_images,
        "final_summary":final_summary
    }
        
    return render(request, 'dashboard.html', context)
    # return HttpResponse("<h1>Hello from Django!</h1>")


@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        try:
            # Parse user message
            data = json.loads(request.body)
            user_input = data.get("message", "")

            greeting_model_output = generate_greet_output(user_input)
            greeting_model_output = json.loads(greeting_model_output)

            if greeting_model_output["is_greeting"]==False:

                # ✅ Load dataset from session
                uploaded_json = request.session.get('uploaded_data')
                if uploaded_json:
                    df = pd.read_json(uploaded_json)
                    # Remove special characters from object columns only
                    df = clean_special_chars(df)
                    #Convert eligible object columns to numerics
                    df = convert_string_numerics_fast(df)
                    metadata = extract_metadata(df)
                elif 'csv_data' in request.session:
                    df = pd.read_json(request.session["csv_data"])
                    metadata = request.session["metadata"]
                    # df = pd.read_json(request.session["csv_data"])
                else:
                    return JsonResponse({'reply': "No dataset found. Please upload a CSV first."})

                # df = df.applymap(remove_special_chars)
                # df = convert_string_numerics(df)
                # metadata = extract_metadata(df)

                # ✅ Load and update chat history from session
                chat_history = request.session.get("chat_history", [])
                chat_history.append({"role": "user", "content": user_input})

                # ✅ Keep only the last 5 interactions
                recent_history = chat_history[-4:]

                # ✅ Build conversation context
                history_prompt = ""
                for item in recent_history:
                    prefix = "User:" if item["role"] == "user" else "Assistant:"
                    history_prompt += f"{prefix} {item['content']}\n"

                # ✅ Combine metadata, history, and latest question
                python_prompt = f"""Here is the previous conversation history between the user and assistant: --- Chat History Start ---
                                    {history_prompt.strip()}
                                    --- Chat History End ---
                                    Now, answer the current user question using the above history **if it's relevant**. Otherwise, answer based on the metadata below and the current question alone.
                                    User Question: {user_input}

                                    Metadata:
                                    {metadata}
                                    """

                result = None
                last_model2_error = ""

                for attempt in range(5):
                    try:
                        retry_prompt_2 = (
                            f"{python_prompt}\n\nPrevious Output error (if any): {last_model2_error}"
                            if last_model2_error else python_prompt
                        )

                        logging.info(f"=== Chatbot TODO model input ===\n\n{python_prompt}")

                        code_response = generate_python_code(retry_prompt_2)
                        logging.info(f"=== Chatbot Python code output ===\n\n{code_response}")

                        code_response = extract_json_from_response(code_response)
                        python_code = fix_llm_code(code_response)

                        local_vars = {"df": df.copy()}
                        exec(python_code, {}, local_vars)
                        result = local_vars.get("result")

                        if result is not None:
                            break
                    except Exception as e:
                        last_model2_error = python_code + str(e)

                if not isinstance(result, (pd.DataFrame, pd.Series)):
                    summary_prompt = f"""{history_prompt.strip()}
                                        User Query: {user_input}
                                        Output Value:
                                        {result}
                                        """
                else:
                    summary_prompt = f"""{history_prompt.strip()}
                                        User Query: {user_input}
                                        Dataset:
                                        {result.to_markdown()}
                                        """
                    
                logging.info(f"=== Chatbot TODO summary_prompt ===\n\n{summary_prompt}")

                # summary_prompt += f"\n\nMetadata:\n{metadata}"
                summary_result = generate_summary(summary_prompt)

                # ✅ Add assistant reply to history
                chat_history.append({"role": "assistant", "content": summary_result})
                request.session["chat_history"] = chat_history

                return JsonResponse({'reply': summary_result})
            else:
                return JsonResponse({'reply': greeting_model_output["greeting"]})


        except Exception as e:
            return JsonResponse({'reply': f"Error: {str(e)}"}, status=400)

@csrf_exempt
def upload_dataset(request):
    if request.method == 'POST':
        file = request.FILES.get('file')
        if not file:
            return JsonResponse({'status': 'No file uploaded.'}, status=400)

        try:
            df = pd.read_csv(file)
            request.session['uploaded_data'] = df.to_json()
            request.session.modified = True
            return JsonResponse({'status': f'Dataset \"{file.name}\" uploaded successfully.'})
        except Exception as e:
            return JsonResponse({'status': f'Failed to upload: {str(e)}'}, status=400)
