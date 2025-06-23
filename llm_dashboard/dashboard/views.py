from django.shortcuts import render
from .llm_pipeline import break_into_subquestions, generate_python_code, generate_plot_code, generate_summary, generate_final_summary,generate_title
import pandas as pd
import numpy as np
import requests
import time
import json
import openai

import difflib
import re

import logging
import pandas as pd

import io
from PIL import Image
import matplotlib.pyplot as plt
import base64

from fpdf import FPDF
import tempfile

from datetime import datetime
import os
# from django.http import HttpResponse

def extract_metadata(df):
    sample = df.head(20).copy()
    
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

# Function to remove unwanted special characters except ., space, and -
def remove_special_chars(s):
    if isinstance(s, str):
        return re.sub(r"[^A-Za-z0-9.\s\-]", "", s)
    return s

# Sample: assume df is your DataFrame
def convert_string_numerics(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try converting to numeric, coerce errors (e.g., '£123.45' or '57.85%')
            cleaned_col = (
                df[col]
                .astype(str)
                # .str.replace(r'[£,%]', '', regex=True)
                .str.strip()
            )
            converted = pd.to_numeric(cleaned_col, errors='coerce')

            # Only update if it actually results in valid numeric values
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

import re

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


# with open("no_plot.png", "rb") as f:
#     NO_PLOT_PLACEHOLDER = base64.b64encode(f.read()).decode("utf-8")

# NO_PLOT_PLACEHOLDER = Image.open("no_plot.png")

import logging
# Configure logging
logging.basicConfig(
    filename="model_outputs.log",  # log file name
    level=logging.INFO,            # log level (can use DEBUG for more granularity)
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def inject_plot_formatting(code: str, height: int = 300) -> str:
    """
    Inject layout and HTML rendering into Plotly figure code.

    Parameters:
    - code (str): Python code string containing the Plotly figure.
    - height (int): Desired plot height.

    Returns:
    - str: Modified Python code with layout and HTML export.
    """
    lines = code.strip().splitlines()
    modified_lines = []

    fig_assigned = False
    for line in lines:
        modified_lines.append(line)
        if not fig_assigned and line.strip().startswith("fig = "):
            # Insert formatting after first fig assignment
            modified_lines.append(
                f"fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height={height})"
            )
            modified_lines.append(
                'plot_html = fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})'
            )
            fig_assigned = True

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

def dashboard_view(request):
    context = {}

    if request.method == 'POST':
        user_question = request.POST.get('user_query', '')

        # Detect fresh session or first use
        if not request.session.get("initialized"):
            request.session["past_questions"] = []
            request.session["initialized"] = True
            
        # ✅ Retrieve the existing list from session or initialize empty list
        past_questions = request.session.get("past_questions", [])

        # ✅ Add the new question if it's not empty
        if user_question.strip():
            past_questions.append(user_question)
            request.session["past_questions"] = past_questions  # ✅ Save back to session

        # ✅ Safely get the file or None
        csv_file = request.FILES.get('csv_file')

        if csv_file:
            # Step 1: Read and store CSV in session
            df = pd.read_csv(csv_file)
            request.session["csv_data"] = df.to_json()
            request.session["columns"] = list(df.columns)

        elif "csv_data" in request.session:
            # Step 2: Load existing CSV from session
            df = pd.read_json(request.session["csv_data"])
        else:
            df = None

            

        # df = pd.read_csv(csv_file)

        df = df.applymap(remove_special_chars)
        df = convert_string_numerics(df)
        metadata = extract_metadata(df)

        user_query = f"User Question: {user_question}\n\nMetadata:\n{metadata}"
        # metadata = {
        #     'columns': list(df.columns),
        #     'dtypes': df.dtypes.astype(str).to_dict(),
        #     'sample': df.head(3).to_dict(orient='records')
        # }


        logging.info(f"=== user_query ===\n\n{user_query}")
        sub_questions = break_into_subquestions(user_query)
        logging.info(f"=== Break down questions ===\n\n{sub_questions}")

        summaries, plot_paths, filtered_data, q_title, filter_result = [], [], [], [], [None,None,None,None,None]

        for i, sub_q in enumerate(sub_questions):
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

                    # filtered_data[i] = result
                    if result is not None:
                        break
                except Exception as e:
                    last_model2_error = python_code+str(e)

            
            # filtered_df = execute_code(code, df)  # safe exec
            # filtered_data.append(filtered_df.head(5).values.tolist())

            # if isinstance(result, (pd.DataFrame, pd.Series)) and result is not None and not result.empty:
            #     result1 = [result.columns.tolist()] + result.values.tolist()
            #     filtered_data.append(result1)
            # elif result is not None:
            #     filtered_data.append(result)
            # else:
            filtered_data.append(None)
                # table_data = None
            
            

            plot_image = None
            if isinstance(result, (pd.DataFrame, pd.Series)):
                # if isinstance(result, (pd.DataFrame)):
                #     result = safe_reset_index(result)
                # else:
                #     result = pd.DataFrame(result).reset_index()

                result = safe_reset_index(result)
                index_cols = ['index', 'Unnamed: 0']
                for col in index_cols:
                    if col in result.columns:
                        result = result.drop(columns=col)

                # local_vars["result"] = result.copy()
                local_vars = {"df": result.copy()}
                # logging.info(f"=== MODEL 3 Dashboard | Before extract_metadata function (Q{i+1}) ===\n\n")
                # result_metadata = extract_metadata(result)
                # logging.info(f"=== MODEL 3 Dashboard | after extract_metadata function (Q{i+1}) ===\n\n")

                summary_prompt = f"User Query: {sub_q}\nDataset: \n{result.to_markdown(index=False)}"
                # summary_prompt = f"Metadata: \n{result_metadata}"
                
            elif i < 5:
                summary_prompt = f"User Query: {sub_q}\nOutput Value: \n{result}"
                if isinstance(result, (int, float)):
                    result = round(result, 2)
                    filter_result[i] = result
                    # filter_result.append(result)
                else:
                    filter_result[i] = result
            else:
                summary_prompt = f"User Query: {sub_q}\nOutput Value: \n{result}"
                    # filter_result.append(result)

                # if result.shape[0] > 1 and result.shape[1] > 1:

            last_model3_error = ''
            viz_code = ''
            for attempt in range(3):
                try:

                    retry_summary_prompt = (
                        f"{summary_prompt}\n\nPrevious Output error (if any): {last_model3_error}"
                        if last_model3_error else summary_prompt
                    )
                    # viz_prompt = f"""Dataset:\n{result.to_markdown(index=False)}"""
                    logging.info(f"=== MODEL 3 Dashboard input (Q{i+1}, Attempt {attempt+1}) ===\n{retry_summary_prompt}\n\n")
                    if isinstance(result, (pd.DataFrame, pd.Series)) and i > 4:
                        viz_code_response = generate_plot_code(retry_summary_prompt)
                        logging.info(f"=== MODEL 3 Dashboard Plot Code (Q{i+1}, Attempt {attempt+1}) ===\n{viz_code_response}\n\n")
                        viz_code_response = extract_json_from_response(viz_code_response)
                        viz_code_response = inject_plot_formatting(viz_code_response, height=280)
                        exec(viz_code_response, {}, local_vars)

                        # Extract HTML from variable (assume model always uses `plot_html`)
                        plot_html = local_vars.get("plot_html", "")
                        plot_paths.append(plot_html)
                        # viz_code_response = query_llm(retry_summary_prompt, MODEL_3_SYSTEM_PROMPT)
                    else:
                        viz_code_response = generate_title(sub_q)
                        logging.info(f"=== MODEL title Dashboard Output (Q{i+1}, Attempt {attempt+1}) ===\n{viz_code_response}\n\n")
                        q_title.append(viz_code_response)
                        logging.info(f"=== MODEL title Dashboard Output Saved")
                        # viz_code_response = query_llm(retry_summary_prompt, MODEL_NO_DF_SYSTEM_PROMPT)
                    
                    # plot_paths.append(encoded_img)
                    break
                except Exception as e:
                    last_model3_error = viz_code_response+str(e)
                    logging.warning(f"⚠️ MODEL 3 Attempt {attempt+1} failed: {e}")
                    plt.close()

            # logging.info(f"=== plot_image (Q{i+1}) ===\n{plot_image}\n\n")

            # plot_paths.append(encoded_img)

            summary_result = generate_summary(summary_prompt)

            # summaries[i] = viz_code_response
            summaries.append(summary_result)

            # viz_code_response = generate_final_summary(retry_summary_prompt)

            
            # plot_path = f'static/plots/subq{i}.png'
            # execute_code(plot_code, filtered_df, save_path=plot_path)
            # plot_paths.append(plot_path)

        logging.info(f"=== MODEL summary Dashboard Input ===\n{summaries}\n\n")
        combined_insights = "\n\n".join(summaries)
        combined_insights = f"What are the most important insights or anomalies?\n\n{combined_insights}"
        final_summary = generate_final_summary(combined_insights)
        # Convert string to Python dict
        final_summary = json.loads(final_summary)

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



        context = {
            "past_questions": past_questions,  # list of previous questions
            "top_5_insights": [       # 5 dicts with value + label
                {"label": q_title[0], "value": filter_result[0]},
                {"label": q_title[1], "value": filter_result[1]},
                {"label": q_title[2], "value": filter_result[2]},
                {"label": q_title[3], "value": filter_result[3]},
                {"label": q_title[4], "value": filter_result[4]}
            ],
            "plot_images": plot_paths,
            "final_summary":final_summary
        }
        
    return render(request, 'dashboard.html', context)
    # return HttpResponse("<h1>Hello from Django!</h1>")
