import os
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# ------------ SETUP ------------
# 1. Initialize Flask App
app = Flask(__name__)

# 2. Enable CORS (Cross-Origin Resource Sharing)
# This allows your frontend (running on a different "origin") to send requests to this backend.
CORS(app)

# 3. Load your Google Gemini API Key from an environment variable for security.
# IMPORTANT: Never hardcode your API key directly in the code.
API_KEY = os.environ.get("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"


# ------------ HELPER FUNCTION ------------
def call_gemini_api(prompt_text):
    """
    Calls the Google Gemini API with a given prompt and returns the response.
    """
    if not API_KEY:
        print("ERROR: GEMINI_API_KEY environment variable not set.")
        return {"error": "Server configuration error: API key is missing."}, 500

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}]
    }
    headers = {'Content-Type': 'application/json'}

    try:
        # Make the POST request to the Gemini API
        response = requests.post(GEMINI_API_URL, json=payload, headers=headers)
        # Raise an exception if the request returned an unsuccessful status code (like 4xx or 5xx)
        response.raise_for_status()
        return response.json(), 200
    except requests.exceptions.RequestException as e:
        # Handle network errors or bad responses from the API
        print(f"Error calling Gemini API: {e}")
        error_content = e.response.text if e.response else "No response from server."
        return {"error": "Failed to communicate with the generative AI model.", "details": error_content}, 500


# ------------ API ENDPOINTS ------------
# This single endpoint will handle all requests from the frontend.
# The frontend will be responsible for creating the specific prompt needed for each task.
@app.route('/api/generate', methods=['POST'])
def handle_generation():
    """
    Receives a prompt from the frontend, sends it to Gemini, and returns the result.
    """
    # Get the JSON data from the request body
    data = request.json
    prompt = data.get('prompt')

    # Ensure a prompt was provided
    if not prompt:
        return jsonify({"error": "A 'prompt' is required in the request body."}), 400

    # Call the helper function to interact with the Gemini API
    result, status_code = call_gemini_api(prompt)
    return jsonify(result), status_code


# ------------ RUN THE APP ------------
if __name__ == '__main__':
    # Run the Flask app on localhost, port 5001
    # debug=True allows the server to auto-reload when you save changes.
    app.run(port=5001, debug=True)