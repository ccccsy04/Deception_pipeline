import time
import random
from openai import OpenAI


API_KEY = 'sk-8FqqW6C8e30lDsNyTaq78eMCoMR01PYnW1BYxFRkPAf1ZOvs' 
MAX_RETRIES = 3
INITIAL_RETRY_DELAY = 1
MAX_RETRY_DELAY = 10



def gpt_api_call(messages, model="gpt-4o"):
    client = OpenAI(api_key=API_KEY, base_url="https://pro.xiaoai.plus/v1")

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                #max_tokens=1000
            )
            content = response.choices[0].message.content.strip()

            # 提取 token 消耗信息
            if hasattr(response, "usage"):
                usage = response.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
                print(f"[Token usage] prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens}")
            else:
                print("[Token usage] not available in response")

            return content
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                delay = min(INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1), MAX_RETRY_DELAY)
                time.sleep(delay)
            else:
                return f"Error after {MAX_RETRIES} attempts: {str(e)}"
                