from openai import OpenAI
import json
import time

class GPT():
    def __init__(self, model_name="gpt-4o", api_key=None, timeout=60):
        self.client = OpenAI(api_key=api_key, timeout=timeout)
        self.model_name = model_name
        self.timeout = timeout

    def call_text(self, text_prompt, tools=None, max_iterations=5):
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
                if self.model_name == "gpt-4o":
                    response = self.client.chat.completions.create(
                        model=self.model_name, 
                        messages=messages,
                        max_tokens=1500,
                        tools=tools if tools else [],
                        tool_choice='required' if tools else None
                    )
                elif self.model_name == "o4-mini":
                    response = self.client.chat.completions.create(
                        model=self.model_name, 
                        messages=messages,
                        max_completion_tokens=1500,
                        tools=tools if tools else [],
                        tool_choice='required' if tools else None
                    )
                tool_calls = response.choices[0].message.tool_calls

                if tool_calls:
                    tool_call = tool_calls[0] 
                    return {
                        "message": response.choices[0].message.content,
                        "function": {
                            "name": tool_call.function.name,
                            "input": json.loads(tool_call.function.arguments)
                        },
                        "usage": {
                            "input": response.usage.prompt_tokens,
                            "output": response.usage.completion_tokens,
                            "total": response.usage.total_tokens
                        }
                    }
                else:
                    return {
                        "message": response.choices[0].message.content,
                        "function": None,
                        "usage": {
                            "input": response.usage.prompt_tokens,
                            "output": response.usage.completion_tokens,
                            "total": response.usage.total_tokens
                        }
                    }
            except:
                time.sleep(0.1)
        raise Exception("OpenAI API call failed after multiple attempts.")
    
    def call_text_images(self, text_prompt, imgs, tools=None, max_iterations=5, pre_knowledge=None):
        messages=  []

        if pre_knowledge is not None:
            messages.append({
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": pre_knowledge
                    }
                ]
            })

        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": text_prompt
                }
            ]
        }   

        for img in imgs:
            message["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img}"
                }
            })

        messages.append(message)

        for i in range(max_iterations):
            try:
                if self.model_name == "gpt-4o":
                    response = self.client.chat.completions.create(
                        model=self.model_name, 
                        messages=messages,
                        max_tokens=1500,
                        tools=tools if tools else [],
                        tool_choice='required' if tools else None
                    )
                elif self.model_name == "o4-mini":
                    response = self.client.chat.completions.create(
                        model=self.model_name, 
                        messages=messages,
                        max_completion_tokens=1500,
                        tools=tools if tools else [],
                        tool_choice='required' if tools else None
                    )
                tool_calls = response.choices[0].message.tool_calls

                if tool_calls:
                    tool_call = tool_calls[0] 
                    return {
                        "message": response.choices[0].message.content,
                        "function": {
                            "name": tool_call.function.name,
                            "input": json.loads(tool_call.function.arguments)
                        },
                        "usage": {
                            "input": response.usage.prompt_tokens,
                            "output": response.usage.completion_tokens,
                            "total": response.usage.total_tokens
                        }

                    }
                else:
                    return {
                        "message": response.choices[0].message.content,
                        "function": None,
                        "usage": {
                            "input": response.usage.prompt_tokens,
                            "output": response.usage.completion_tokens,
                            "total": response.usage.total_tokens
                        }
                    }
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(0.1)
        raise Exception("OpenAI API call failed after multiple attempts.")