from .llm_utils import query_llm
from .system_ins import MODEL_DATA_IDENTIFIER, MODEL_ANOMALIES_SYSTEM_PROMPT, MODEL_TABLE_SCHEMA_SYSTEM_PROMPT, MODEL_GREETING_PROMPT, MODEL_CLASSIFICATION_SYSTEM_PROMPT, MODEL_DATA_PROCESSING_SYSTEM_PROMPT, MODEL_VIZ_SYSTEM_PROMPT, MODEL_NO_DF_SYSTEM_PROMPT, MODEL_SUMMARY_SYSTEM_PROMPT, MODEL_BREAKDOWN_SYSTEM_PROMPT, MODEL_FINAL_SUMMARY_PROMPT, MODEL_TITLE_SUMMARY_PROMPT, MODEL_DESCRIPTION_SYSTEM_PROMPT
import json


def anomalies_agent(user_query):
    anomalies_agent_response = query_llm(user_query, MODEL_ANOMALIES_SYSTEM_PROMPT)
    return anomalies_agent_response

def data_identifier_agent(user_query):
    data_identifier_agent_response = query_llm(user_query, MODEL_DATA_IDENTIFIER)
    return data_identifier_agent_response

def table_schema_agent(user_query):
    table_schema_agent_response = query_llm(user_query, MODEL_TABLE_SCHEMA_SYSTEM_PROMPT)
    return table_schema_agent_response

def classification_agent(user_query):
    classification_agent_response = query_llm(user_query, MODEL_CLASSIFICATION_SYSTEM_PROMPT)
    return classification_agent_response

def break_into_subquestions(user_query):
    sub_q_response = query_llm(user_query, MODEL_BREAKDOWN_SYSTEM_PROMPT)
    sub_questions = json.loads(sub_q_response).get("sub_questions", [])
    # Call your LLM (e.g., GPT) and return list of 6 sub-questions
    return sub_questions

def generate_python_code(updated_user_prompt, temp_value = 0.8, model_name = "gpt-3.5-turbo"):
    return query_llm(updated_user_prompt, MODEL_DATA_PROCESSING_SYSTEM_PROMPT, temp_value, model_name)
    # return f"""filtered_df = df.head(10)  # dummy logic"""

def generate_plot_code(updated_user_prompt, temp_value = 0.8, model_name = "gpt-3.5-turbo"):
    return query_llm(updated_user_prompt, MODEL_VIZ_SYSTEM_PROMPT)

def generate_title(updated_user_prompt):
    return query_llm(updated_user_prompt, MODEL_TITLE_SUMMARY_PROMPT)

def generate_summary(updated_user_prompt):
    return query_llm(updated_user_prompt, MODEL_SUMMARY_SYSTEM_PROMPT)

def generate_description(updated_user_prompt):
    return query_llm(updated_user_prompt, MODEL_DESCRIPTION_SYSTEM_PROMPT)

def generate_final_summary(updated_user_prompt):
    return query_llm(updated_user_prompt, MODEL_FINAL_SUMMARY_PROMPT)

def generate_greet_output(updated_user_prompt):
    return query_llm(updated_user_prompt, MODEL_GREETING_PROMPT,0.8,"gpt-4-turbo")

# def generate_final_result(updated_user_prompt):
#     return query_llm(updated_user_prompt, MODEL_FINAL_SUMMARY_PROMPT)

# def execute_code(code, df, save_path=None):
#     local_env = {'df': df, 'save_path': save_path, 'plt': __import__('matplotlib.pyplot')}
#     exec(code, {}, local_env)
#     return local_env.get('filtered_df', df)
