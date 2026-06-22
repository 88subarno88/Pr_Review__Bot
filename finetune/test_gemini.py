import os
from dotenv import load_dotenv
load_dotenv()
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                      http_options=types.HttpOptions(api_version="v1"))
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in one word.",
)
print("SUCCESS:", resp.text)
