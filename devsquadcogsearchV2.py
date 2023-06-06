#!/usr/bin/python3.8
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import requests
import json
import sys
import uuid
import os
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, AzureChatCompletion
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/v1/*": {"origins": "*", "supports_credentials": True, "max_age": 3600}})
app.debug = os.environ.get('FLASK_DEBUG', False)

# Initialize Semantic Kernel
kernel = sk.Kernel()
deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
kernel.add_chat_service("dv", AzureChatCompletion(deployment, endpoint, api_key))

def read_api_key(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except Exception as e:
        logger.error(f"Failed to read Azure Cognitive Search API key: {e}")
        raise
# Read Azure Cognitive Search API key from a file
azure_search_api_key = read_api_key("azure_search_key.txt")
azure_search_service_name = "<NAME>"



def azure_search_query(search_query_text):
    try:
        url = f"https://{azure_search_service_name}.search.windows.net/indexes/<INDEXNAME>/docs/search?api-version=2021-04-30-Preview"
        payload = {
            "search": search_query_text,
            "top": 3,
        }

        headers = {"api-key": azure_search_api_key, "Content-Type": "application/json"}

        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()["value"]

        # Find the highest scored result
        highest_score = -1
        highest_score_text = ""
        for result in results:
            if result["@search.score"] > highest_score:
                highest_score = result["@search.score"]
                highest_score_text = result["content"]

        return highest_score_text
    except Exception as e:
        logger.error(f"An error occurred while performing Azure Cognitive Search query: {e}")
        raise

def chatgpt_with_azure_search(search_query_text):
    triggers = ["Dev Squad", "CSU"]

    if any(trigger in search_query_text for trigger in triggers):
        try:
            search_response = azure_search_query(search_query_text)
            prompt = f"Using the following text from an Azure Cognitive Search query: '{search_response}', please generate a well-informed and human-like answer for the question: '{search_query_text}'. If demanded in the question, use your knowledge based on previous training and also check on available Internet sources to create the most complete answer."
        except Exception as e:
            logger.error(f"Failed to perform Azure Cognitive Search query: {e}")
            return None
    else:
        prompt = """{{$input}}"""

    try:
        sk_devsquad_function = kernel.create_semantic_function(prompt, max_tokens=1000, temperature=0.7, top_p=0.5)
        chatgpt_response = sk_devsquad_function(search_query_text)
        return chatgpt_response
    except Exception as e:
        logger.error(f"An error occurred while generating chat response: {e}")
        raise

@app.route('/api/v1/chat', methods=['POST', 'OPTIONS'])
@cross_origin()
def chat():
    try:
        data = request.get_json()
        input_text = data.get('input_text')
        if not input_text:
            return jsonify({'error': 'Invalid request payload'}), 400

        response = chatgpt_with_azure_search(input_text)
        if response:
            return jsonify({'response': str(response)})
        else:
            return jsonify({'response': 'No data'}), 204
    except Exception as e:
        logger.error(f"An error occurred during chat request: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    try:
        # Load Azure Cognitive Search API key
        azure_search_api_key = read_api_key("azure_search_key.txt")

        # Start the Flask application
        app.run()
    except Exception as e:
        logger.error(f"Failed to start the application: {e}")

       
