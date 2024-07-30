from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import threading
import argparse
import time

app = Flask(__name__)
CORS(app)

# Load the model and tokenizer
model_name = "dolphin-2.9-llama3-8b"  # Replace with the actual model name
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Ensure the model is on the correct device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def prepare_input(text):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs

def predict(text):
    inputs = prepare_input(text)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=50,  # Adjust max_length as needed
            temperature=0.7,  # Sampling temperature
            top_p=0.9,  # Nucleus sampling
            top_k=50,  # Top-k sampling
            num_return_sequences=1  # Number of sequences to return
        )
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.json
    text = data.get('text', '')
    result = predict(text)
    return jsonify({'prediction': result})

def run_flask():
    app.run(host='0.0.0.0', port=5000)

def get_prediction(text):
    url = "http://localhost:5000/predict"
    payload = {"text": text}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json().get("prediction")
    else:
        return f"Error: {response.status_code}"

def cli_interface():
    time.sleep(1)  # Ensure the server has time to start
    while True:
        user_input = input("Enter your message (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        prediction = get_prediction(user_input)
        print("AI Response:", prediction)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Flask server with CLI interface.")
    parser.add_argument('--cli', action='store_true', help="Run the CLI interface")
    args = parser.parse_args()

    if args.cli:
        # Start the Flask server in a separate thread
        flask_thread = threading.Thread(target=run_flask)
        flask_thread.daemon = True
        flask_thread.start()

        # Run the CLI interface
        cli_interface()
    else:
        # Just run the Flask server
        run_flask()
