import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
MODEL_NAME = 'gemini-2.5-flash-lite'

print('API_KEY present:', bool(API_KEY))
try:
    genai.configure(api_key=API_KEY)
    client = genai.GenerativeModel(MODEL_NAME)
    resp = client.generate_content('Hello from smoke test')
    print('Model response:', resp.text[:200])
except Exception as e:
    print('Error contacting Gemini:', e)
