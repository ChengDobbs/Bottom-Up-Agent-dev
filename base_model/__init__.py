from .gpt import GPT
from .claude import Claude
from dotenv import load_dotenv
import os


def creat_base_model(model_name, timeout=60):
    print(f"Model name: {model_name}")
    load_dotenv()
    if model_name == "gpt-4o" or model_name == "gpt-4o-vision" or model_name == "o4-mini":
        return GPT(model_name=model_name, api_key=os.getenv("OPENAI_API_KEY"))
    if model_name == "claude-3-7-sonnet-20250219":
        return Claude(model_name=model_name, api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=timeout)
    if model_name == "UI_TARS":
        from .UI_TARS import UI_TARS
        return UI_TARS(model_name, api_key=os.getenv("UI_TARS_API_KEY"))
    else:
        print(f"Model {model_name} not found")
        return None