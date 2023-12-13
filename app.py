from flask import Flask, render_template, request, jsonify
import configparser
from llamaapi import LlamaAPI
import html
import json

app = Flask(__name__)

config = configparser.ConfigParser()
config.read('token.ini')

llama_token = config['llama']['token']

llama = LlamaAPI(llama_token)

conversation_context = []

@app.route('/')
def home():
    with open('data.json', 'r', encoding='utf-8') as file:
        data_json_content = file.read()
    return render_template('index.php', data_json_content=data_json_content)

@app.route('/get')
def get_bot_response():
    user_text = request.args.get('msg')
    llama2_response = call_llama2_api(user_text)
    bot_response = process_llama2_response(llama2_response)
    conversation_context.append({"user_input": user_text, "bot_response": bot_response})
    return jsonify({"bot_response": bot_response})

def call_llama2_api(user_input):    
    api_request_json = {
        "messages": [
            {"role": "user", "content": user_input},
        ],
        "functions": [
        ],
        "stream": False,
    }
    response = llama.run(api_request_json)
    print(f"Llama2 Response: {response.text}")
    return response

def process_llama2_response(llama2_response):
    try:
        data = llama2_response.json()
        if 'choices' in data and data['choices']:
            assistant_message = data['choices'][0]['message']['content']
            assistant_message_escaped = html.escape(assistant_message)
            assistant_message = assistant_message_escaped.replace('\n', '<br>')
            return assistant_message     
        else:
            return "Je ne peux pas fournir d'informations pour le moment. Veuillez réessayer plus tard."
    except Exception as e:
        return "Une erreur s'est produite lors du traitement de la réponse."

if __name__ == '__main__':
    app.run(debug=True)
    