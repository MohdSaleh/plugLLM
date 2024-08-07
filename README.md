# PlugLLM

PlugLLM is a flexible and easy-to-use library for integrating various Language Learning Models (LLMs) into your projects. It provides a unified interface for working with different LLM providers, making it simple to switch between models or use multiple models in the same project.

## Features

- Support for multiple LLM providers (OpenAI, Groq, Fireworks, etc.)
- Easy-to-use API for submitting prompts and receiving responses
- Optimized prompt handling for improved performance
- Flexible system and user message creation

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/plugllm.git
   cd plugllm
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install the LLM client of your choice. For example, for OpenAI:
   ```
   pip install openai
   ```

## Usage

Here's a basic example of how to use PlugLLM with OpenAI's GPT-4:

```python
from plugRAG_module import PlugLLM_Base
from openai import OpenAI

# Initialize the OpenAI client
client = OpenAI(base_url='https://api.openai.com/v1', api_key="your_openai_api_key")

# Choose your LLM model
llm_model = 'gpt-4o'

# Create a PlugLLM instance
plugLLM = PlugLLM_Base.create_plugLLM(llm_model, client)

# Use the optimized prompt method
optimised_response = plugLLM.easy_prompt("Tell me a joke about programming.")

# Use the standard prompt method with system and user messages
normal_response = plugLLM.submit_prompt([
    plugLLM.system_message("You are a helpful assistant."),
    plugLLM.user_message("What are the benefits of using Python for data analysis?")
])

print("Optimised response:", optimised_response)
print("Normal response:", normal_response)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License 

## Contact

If you have any questions or feedback, please open an issue on this GitHub repository.
