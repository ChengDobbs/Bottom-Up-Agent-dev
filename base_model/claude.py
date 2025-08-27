import requests
import json
import anthropic
import time
from anthropic import APITimeoutError, APIConnectionError, RateLimitError

class Claude():
    def __init__(self, model_name="claude-3-7-sonnet-20250219", api_key=None, timeout=60):
        self.client = anthropic.Client(api_key=api_key, timeout=timeout)
        self.model_name = model_name
        self.timeout = timeout
    
    def call_text(self, text_prompt, tools=None, max_iterations=3):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        ]

        for i in range(max_iterations):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=3000,
                    tools=tools if tools else [],
                    timeout=self.timeout
                )

                answer = None
                tool_call = None

                for block in response.content:
                    if block.type == "text" and answer is None:
                        answer = block.text
                    elif block.type == "tool_use":
                        tool_call = {
                            "name": block.name,
                            "input": block.input
                        }

                return {
                    "message": answer,
                    "function": tool_call,
                    "usage": {
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                        "total": response.usage.input_tokens + response.usage.output_tokens
                    }
                }
            except (APITimeoutError, APIConnectionError) as e:
                print(f"Network error on attempt {i+1}/{max_iterations}: {e}")
                if i < max_iterations - 1:
                    time.sleep(min(2 ** i, 10))  # Exponential backoff with max 10s
                else:
                    raise Exception(f"Claude API call failed after {max_iterations} attempts due to network issues: {e}")
            except RateLimitError as e:
                print(f"Rate limit error on attempt {i+1}/{max_iterations}: {e}")
                if i < max_iterations - 1:
                    time.sleep(min(5 * (i + 1), 30))  # Longer wait for rate limits
                else:
                    raise Exception(f"Claude API call failed after {max_iterations} attempts due to rate limiting: {e}")
            except KeyboardInterrupt:
                print("User interrupted the API call")
                raise
            except Exception as e:
                print(f"Unexpected error on attempt {i+1}/{max_iterations}: {e}")
                if i < max_iterations - 1:
                    time.sleep(0.5)
                else:
                    raise Exception(f"Claude API call failed after {max_iterations} attempts: {e}")
        raise Exception("Claude API call failed after multiple attempts.")
    
    
    def call_text_images(self, text_prompt, imgs, tools=None, max_iterations=3, pre_knowledge=None):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }   
        ]

        for img in imgs:
            messages[0]["content"].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img
                }
            })

        
        for i in range(max_iterations):
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    system=pre_knowledge if pre_knowledge else "you are a helpful assistant",
                    messages=messages,
                    max_tokens=3000,
                    tools=tools if tools else [],
                    timeout=self.timeout
                )

                answer = None
                tool_call = None

                for block in response.content:
                    if block.type == "text" and answer is None:
                        answer = block.text
                    elif block.type == "tool_use":
                        tool_call = {
                            "name": block.name,
                            "input": block.input
                        }

                return {
                    "message": answer,
                    "function": tool_call,
                    "usage": {
                        "input": response.usage.input_tokens,
                        "output": response.usage.output_tokens,
                        "total": response.usage.input_tokens + response.usage.output_tokens
                    }
                }
            except (APITimeoutError, APIConnectionError) as e:
                print(f"Network error on attempt {i+1}/{max_iterations}: {e}")
                if i < max_iterations - 1:
                    time.sleep(min(2 ** i, 10))  # Exponential backoff with max 10s
                else:
                    raise Exception(f"Claude API call failed after {max_iterations} attempts due to network issues: {e}")
            except RateLimitError as e:
                print(f"Rate limit error on attempt {i+1}/{max_iterations}: {e}")
                if i < max_iterations - 1:
                    time.sleep(min(5 * (i + 1), 30))  # Longer wait for rate limits
                else:
                    raise Exception(f"Claude API call failed after {max_iterations} attempts due to rate limiting: {e}")
            except KeyboardInterrupt:
                print("User interrupted the API call")
                raise
            except Exception as e:
                print(f"Unexpected error on attempt {i+1}/{max_iterations}: {e}")
                if i < max_iterations - 1:
                    time.sleep(0.5)
                else:
                    raise Exception(f"Claude API call failed after {max_iterations} attempts: {e}")
        raise Exception("Claude API call failed after multiple attempts.")

    


