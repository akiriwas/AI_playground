# curl http://192.168.1.132:1234/v1/chat/completions -H "Content-Type: application/json" -d '{
# "model": "llama-3.2-1b-instruct",
# "messages": [{"role": "user", "content": "Write a python implementation of the battleships game."}],
# "temperature": 0.7
# }'


import openai

# Define the local LM Studio API endpoint
LM_STUDIO_API_URL = "http://192.168.1.132:1234/v1"  # Adjust port if needed

# Set up the OpenAI-compatible client
client = openai.OpenAI(base_url=LM_STUDIO_API_URL, api_key='lm-studio')

# Define the request parameters
response = client.chat.completions.create(
    model="llama-3.2-1b-instruct",  # Adjust the model name based on what's running in LM Studio
    messages=[
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Write a python application demonstrating how to reverse a linked list."}
    ],
    temperature=0.7
)

# Print the model's response
print(response.choices[0].message.content)
