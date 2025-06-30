# dashboard/llm_utils.py

import os
import openai
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def query_llm(prompt, system_prompt, temperature=0.8, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content
