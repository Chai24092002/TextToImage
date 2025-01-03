
import requests

data_set = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-3.5-large"
headers = {"Authorization": "Bearer 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"}

# Function to query the model
def config(payload):
    response = requests.post(data_set, headers=headers, json=payload)
    response.raise_for_status()
    return response.content
