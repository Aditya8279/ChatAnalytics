from django.shortcuts import render
from .llm_pipeline import generate_greet_output, classification_agent, break_into_subquestions, generate_python_code, generate_plot_code, generate_summary, generate_final_summary,generate_title, generate_description
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

import io
from PIL import Image
import matplotlib.pyplot as plt
import base64

from fpdf import FPDF
import tempfile

from datetime import datetime
import os
# from django.http import HttpResponse

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from django.utils.timezone import now

def extract_metadata(df):
    sample = df.head(13).copy()
    
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
        stripped = line.strip()
        
        # Skip fig.show() line
        if stripped.startswith("fig.show()"):
            continue
        
        modified_lines.append(line)

        # Twilight color palette
        twilight_colors = px.colors.cyclical.Twilight

        if not fig_assigned and stripped.startswith("fig = "):
            # Insert formatting after first fig assignment
            modified_lines.append(
                f"fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), autosize=True, height={height}, "
                f"plot_bgcolor='white', paper_bgcolor='white', "
                f"xaxis=dict(showgrid=False, showticklabels=False), "
                f"yaxis=dict(showgrid=True, showticklabels=True, gridcolor='lightgrey'))"
                # f"colorway={twilight_colors})"
            )
            modified_lines.append(
                'plot_html = fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})'
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


@csrf_exempt
def upload_csv(request):
    if request.method == 'POST':
        csv_file = request.FILES.get('csv_file')

        if csv_file:
            # Step 1: Read and store CSV in session
            df = pd.read_csv(csv_file, dtype=str, low_memory=False)
            # Step 1: Drop columns with > 50% missing values
            df = df.loc[:, df.isnull().mean() <= 0.5]

            # Step 2: Fill remaining missing values
            for col in df.columns:
                # if pd.api.types.is_numeric_dtype(df[col]):
                #     df[col] = df[col].fillna(0)
                # else:
                df[col] = df[col].fillna(pd.NA)  # or None

            request.session["csv_data"] = df.to_json()
            request.session["columns"] = list(df.columns)

            # Save to a file for download
            download_path = os.path.join(settings.MEDIA_ROOT, "cleaned_data.csv")
            df.to_csv(download_path, index=False)

            return JsonResponse({
                "message": f"✅ The final dataset contains {df.shape[1]} columns and {df.shape[0]} rows after preprocessing.",
                "download_url": f"{settings.MEDIA_URL}cleaned_data.csv"
            })

        elif "csv_data" in request.session:
            # Step 2: Load existing CSV from session
            df = pd.read_json(request.session["csv_data"])
        else:
            df = None
            return JsonResponse({"error": "Invalid request"}, status=400)

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
    plot_paths=[]
    final_summary=None

    if request.method == 'POST':
        user_question = request.POST.get('user_query', '')

        # ✅ Add the new question if it's not empty
        if user_question.strip():
            # ✅ Retrieve the existing list from session or initialize empty list
            past_questions = request.session.get("past_questions", [])
            # past_questions.append(user_question)
            past_questions.insert(0, user_question)
            request.session["past_questions"] = past_questions  # ✅ Save back to session

        # ✅ Load preprocessed DataFrame from session
        if "csv_data" in request.session:
            df = pd.read_json(request.session["csv_data"])
        else:
            return render(request, "dashboard.html", {
                "error": "❌ No CSV uploaded yet. Please upload a file first.",
                "past_questions": past_questions
            })
        
        df = df.applymap(remove_special_chars)
        df = convert_string_numerics(df)
        metadata = extract_metadata(df)

        analysis_info = classification_agent(user_question)
        logging.info(f"=== classification_agent output ===\n\n{analysis_info}")
        # Parse the string into a dictionary
        analysis_info = json.loads(analysis_info)

        user_query = f"User Question: {user_question}\n\nMetadata:\n{metadata}"


        logging.info(f"=== user_query ===\n\n{user_query}")
        sub_questions = break_into_subquestions(user_query)
        logging.info(f"=== Break down questions ===\n\n{sub_questions}")

        
        # updated_list = insert_user_question(sub_questions, user_question, analysis_info)
        # logging.info(f"=== Updated Break down questions list ===\n\n{updated_list}")

        # sub_questions = updated_list

        summaries, plot_paths, filtered_data, q_title, filter_result, description_list = [], [], [], [None,None,None,None,None,None,None,None,None,None,None,None,None], [None,None,None,None,None,None,None,None,None,None,None,None, None], [None,None,None,None,None,None,None,None,None,None,None,None,None]

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

                    result = safe_reset_index(result)

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
            if isinstance(result, pd.DataFrame) and result.shape[0] > 1 and result.shape[1] > 1 or isinstance(result, pd.Series) and result.shape[0] > 1:
                # if isinstance(result, (pd.DataFrame)):
                #     result = safe_reset_index(result)
                # else:
                #     result = pd.DataFrame(result).reset_index()

                # result = safe_reset_index(result)
                # index_cols = ['index', 'Unnamed: 0']
                # for col in index_cols:
                #     if col in result.columns:
                #         result = result.drop(columns=col)

                for col in result.columns:
                    if is_period_dtype(result[col]):
                        result[col] = result[col].astype(str)

                result_head = result.head(14)

                # local_vars["result"] = result.copy()
                local_vars = {"df": result.copy()}
                # logging.info(f"=== MODEL 3 Dashboard | Before extract_metadata function (Q{i+1}) ===\n\n")
                # result_metadata = extract_metadata(result)
                # logging.info(f"=== MODEL 3 Dashboard | after extract_metadata function (Q{i+1}) ===\n\n")

                # summary_prompt = f"User Query: {sub_q}\nDataset: \n{result_head.to_markdown(index=False)}"
                summary_prompt = f"Dataset: \n{result_head.to_markdown(index=False)}"
                
            else:
                # summary_prompt = f"User Query: {sub_q}\nOutput Value: \n{result}"
                
                # filter_result[i] = result

                if isinstance(result, (int, float)):
                    result = round(result, 2)
                    summary_prompt = f"User Query: {sub_q}\nOutput Value: \n{result}"
                    description = generate_description(summary_prompt)
                    description_list[i] = description
                    filter_result[i] = result
                    # filter_result.append(result)
                else:
                    if not isinstance(result, (pd.DataFrame, pd.Series)):
                        summary_prompt = f"User Query: {sub_q}\nOutput Value: \n{result}"
                        description = generate_description(summary_prompt)
                        description_list[i] = description
                        filter_result[i] = result
                    else:
                        summary_prompt = f"Dataset: \n{result.to_markdown(index=False)}"
                        description = generate_description(summary_prompt)
                        # description_list[i] = description
                        # filter_result[i] = result

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
                    if isinstance(result, pd.DataFrame) and result.shape[0] > 1 and result.shape[1] > 1 or isinstance(result, pd.Series) and result.shape[0] > 1:
                        viz_code_response = generate_plot_code(retry_summary_prompt)
                        logging.info(f"=== MODEL 3 Dashboard Plot Code (Q{i+1}, Attempt {attempt+1}) ===\n{viz_code_response}\n\n")
                        viz_code_response = extract_json_from_response(viz_code_response)
                        viz_code_response = inject_plot_formatting(viz_code_response, height=280)
                        logging.info(f"=== MODEL 3 Dashboard local_vars access (Q{i+1}, Attempt {attempt+1}) ===\n{local_vars}\n\n")
                        exec(viz_code_response, {}, local_vars)

                        # Extract HTML from variable (assume model always uses `plot_html`)
                        plot_html = local_vars.get("plot_html", "")
                        plot_paths.append(plot_html)
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

            # logging.info(f"=== plot_image (Q{i+1}) ===\n{plot_image}\n\n")

            # plot_paths.append(encoded_img)

            summary_result = generate_summary(summary_prompt)

            # summaries[i] = viz_code_response
            # summaries.append(summary_result)
            summaries.append({
                "question": sub_q,
                "summary": summary_result
            })

            # viz_code_response = generate_final_summary(retry_summary_prompt)

            
            # plot_path = f'static/plots/subq{i}.png'
            # execute_code(plot_code, filtered_df, save_path=plot_path)
            # plot_paths.append(plot_path)

        logging.info(f"=== MODEL summary Dashboard Input ===\n{summaries}\n\n")
        # combined_insights = "\n\n".join(summaries)

        combined_insights = "\n\n".join(
            f"Q: {item['question']}\nA: {item['summary']}" for item in summaries
        )
        combined_insights = f"What are the most important insights or anomalies based on User's Question?\n\nUser Question:{user_question}\n\ncombined Q&A:\n{combined_insights}"
        final_summary = generate_final_summary(combined_insights)
        logging.info(f"=== MODEL final summary Dashboard Output ===\n{final_summary}\n\n")
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


        top_insights = []
        for label, value, description in zip(q_title, filter_result, description_list):
            if label and value:  # Exclude if either is None, empty, or falsy
                top_insights.append({"label": label, "value": value, "description": description})

    context = {
        "past_questions": past_questions,  # list of previous questions
        "top_insights": top_insights[:5],
        "plot_images": plot_paths,
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
                elif 'csv_data' in request.session:
                    df = pd.read_json(request.session["csv_data"])
                else:
                    return JsonResponse({'reply': "No dataset found. Please upload a CSV first."})

                df = df.applymap(remove_special_chars)
                df = convert_string_numerics(df)
                metadata = extract_metadata(df)

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
