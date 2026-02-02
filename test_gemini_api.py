# test_gemini_api.py - CORRECTED VERSION

import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

# IMPORTANT: Use the correct, current model name
CORRECT_MODEL_NAME = "gemini-2.5-flash" 

# Ensure the API key is loaded
API_KEY = os.getenv("GEMINI_API_KEY")

try:
    # Initialize the client
    client = genai.Client(api_key=API_KEY)
    print("✅ Gemini client initialized successfully!")

    # Attempt to generate content with the CORRECT model name
    response = client.models.generate_content(
        model=CORRECT_MODEL_NAME,  # <--- FIX IS HERE
        contents="Hello, LegalBot! What is 2 + 2?"
    )

    print("\nResponse from Gemini:")
    print(response.text)

except Exception as e:
    # This should now return the correct 429 error if you hit the rate limit, 
    # but NOT the 404 error.
    print(f"❌ Error: {e}")