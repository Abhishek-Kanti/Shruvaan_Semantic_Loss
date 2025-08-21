#currently not in use as of 2023-10-30
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

def llm_client(prompt: str) -> str:
    """
    Wrapper around Gemini to return text output given a prompt.
    """
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text.strip()
