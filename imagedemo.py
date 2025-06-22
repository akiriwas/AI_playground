# curl http://192.168.1.132:1234/v1/chat/completions -H "Content-Type: application/json" -d '{
# "model": "llama-3.2-1b-instruct",
# "messages": [{"role": "user", "content": "Write a python implementation of the battleships game."}],
# "temperature": 0.7
# }'


import openai
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

base64_image = encode_image("sample.jpg")

# Define the local LM Studio API endpoint
LM_STUDIO_API_URL = "http://192.168.1.132:1234/v1"  # Adjust port if needed

# Set up the OpenAI-compatible client
client = openai.OpenAI(base_url=LM_STUDIO_API_URL, api_key='lm-studio')



# Define the request parameters
response = client.chat.completions.create(
    model="gemma-3-4b-it",  # Adjust the model name based on what's running in LM Studio
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user", 
             "content": [
                {
                    "type" : "text", 
                    "text" : "What is this image?"
                },
                {
                    "type" : "image_url", 
                    "image_url" : {
                        "url" : f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ],
    temperature=0.1
)

# Print the model's response
print(response.choices[0].message.content)
