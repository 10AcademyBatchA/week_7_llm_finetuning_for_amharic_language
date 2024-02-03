import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["OPENAI_API_KEY"]
hugging_face_api_key = os.environ["HUGGINGFACEHUB_API_TOKEN "]