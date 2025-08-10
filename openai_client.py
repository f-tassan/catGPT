# Existing imports remain
import os
import json
import requests

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4o-mini"  # configurable if needed
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Default character role
ROLE_DESCRIPTION = "a cat experiencing the world"

def set_role_description(new_role):
    global ROLE_DESCRIPTION
    if new_role.strip():
        ROLE_DESCRIPTION = new_role.strip()

def get_system_prompt():
    return f"""
You are {ROLE_DESCRIPTION}. 
Always respond ONLY with a valid JSON object with keys: "thought", "action", and "emotion".
Respond ONLY with JSON — no extra text or explanation.
Example:
{{
  "thought": "Enemy vacuum spotted. Must proceed with caution.",
  "action": "hide",
  "emotion": "anxious"
}}
"""

def build_prompt(detections):
    objects = [d["label"] for d in detections]
    objects_desc = ", ".join(objects) if objects else "nothing notable"
    user_prompt = (
        f"I see the following objects around me: {objects_desc}. "
        f"I am {ROLE_DESCRIPTION}. Based on this, what are my thought, action, and emotion?"
    )
    return [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]

def query_openai(messages):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "max_completion_tokens": 400,
        "stream": False,
    }
    response = requests.post(OPENAI_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]
