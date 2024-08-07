import os

from openai import OpenAI
from openai import APIError, InternalServerError

from ..base import PlugBase


class OpenAI_Chat(PlugBase):
    def __init__(self, client=None, config=None):
        PlugBase.__init__(self, config=config)

        # default parameters - can be overrided using config
        self.temperature = 0.7
        self.max_tokens = 500

        if "temperature" in config:
            self.temperature = config["temperature"]

        if "max_tokens" in config:
            self.max_tokens = config["max_tokens"]

        if "api_type" in config:
            raise Exception(
                "Passing api_type is now deprecated. Please pass an OpenAI client instead."
            )

        if "api_base" in config:
            raise Exception(
                "Passing api_base is now deprecated. Please pass an OpenAI client instead."
            )

        if "api_version" in config:
            raise Exception(
                "Passing api_version is now deprecated. Please pass an OpenAI client instead."
            )

        if client is not None:
            self.client = client
            return

        if config is None and client is None:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            return

        if "api_key" in config:
            self.client = OpenAI(api_key=config["api_key"])

    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}

    # def submit_prompt(self, prompt, **kwargs) -> str:
    #     if prompt is None:
    #         raise Exception("Prompt is None")

    #     if len(prompt) == 0:
    #         raise Exception("Prompt is empty")

    #     # Count the number of tokens in the message log
    #     # Use 4 as an approximation for the number of characters per token
    #     num_tokens = 0
    #     for message in prompt:
    #         num_tokens += len(message["content"]) / 4

    #     if kwargs.get("model", None) is not None:
    #         model = kwargs.get("model", None)
    #         print(
    #             f"Using model {model} for {num_tokens} tokens (approx)"
    #         )
    #         response = self.client.chat.completions.create(
    #             model=model,
    #             messages=prompt,
    #             max_tokens=self.max_tokens,
    #             stop=None,
    #             temperature=self.temperature,
    #         )
    #     elif kwargs.get("engine", None) is not None:
    #         engine = kwargs.get("engine", None)
    #         print(
    #             f"Using model {engine} for {num_tokens} tokens (approx)"
    #         )
    #         response = self.client.chat.completions.create(
    #             engine=engine,
    #             messages=prompt,
    #             max_tokens=self.max_tokens,
    #             stop=None,
    #             temperature=self.temperature,
    #         )
    #     elif self.config is not None and "engine" in self.config:
    #         print(
    #             f"Using engine {self.config['engine']} for {num_tokens} tokens (approx)"
    #         )
    #         response = self.client.chat.completions.create(
    #             engine=self.config["engine"],
    #             messages=prompt,
    #             max_tokens=self.max_tokens,
    #             stop=None,
    #             temperature=self.temperature,
    #         )
    #     elif self.config is not None and "model" in self.config:
    #         print(
    #             f"Using model {self.config['model']} for {num_tokens} tokens (approx)"
    #         )
    #         response = self.client.chat.completions.create(
    #             model=self.config["model"],
    #             messages=prompt,
    #             max_tokens=self.max_tokens,
    #             stop=None,
    #             temperature=self.temperature,
    #         )
    #     else:
    #         if num_tokens > 3500:
    #             model = "gpt-3.5-turbo-16k"
    #         else:
    #             model = "gpt-3.5-turbo"

    #         print(f"Using model {model} for {num_tokens} tokens (approx)")
    #         response = self.client.chat.completions.create(
    #             model=model,
    #             messages=prompt,
    #             max_tokens=self.max_tokens,
    #             stop=None,
    #             temperature=self.temperature,
    #         )

    #     # Find the first response from the chatbot that has text in it (some responses may not have text)
    #     for choice in response.choices:
    #         if "text" in choice:
    #             return choice.text

    #     # If no response with text is found, return the first response's content (which may be empty)
    #     return response.choices[0].message.content
    def submit_prompt(self, prompt, **kwargs) -> str:
        if prompt is None or len(prompt) == 0:
            raise ValueError("Prompt is None or empty")

        num_tokens = sum(len(message["content"]) // 4 for message in prompt)

        try:
            model = kwargs.get("model") or self.config.get("model") or ("gpt-3.5-turbo-16k" if num_tokens > 3500 else "gpt-3.5-turbo")
            print(f"Using model {model} for {num_tokens} tokens (approx)")

            response = self.client.chat.completions.create(
                model=model,
                messages=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            # The new API returns the content directly in the message
            return response.choices[0].message.content

        except InternalServerError as e:
            print(f"Internal Server Error: {str(e)}")
            raise
        except APIError as e:
            print(f"API Error: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            raise
