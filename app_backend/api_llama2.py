import os
import sys
import argparse
from flask import Flask, jsonify, request, Response
import urllib.parse
import requests
import time
import json
from dotenv import load_dotenv

load_dotenv()
MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(MAIN_DIR)
API_KEY = os.environ.get('API_KEY')
REPHRASED_TOKEN = os.environ.get('REPHRASED_TOKEN')

app = Flask(__name__)

parser = argparse.ArgumentParser(description="An example of using server.cpp with a similar API to OAI. It must be used together with server.cpp.")
parser.add_argument("--stop", type=str, help="the end of response in chat completions(default: '</s>')", default="</s>")
parser.add_argument("--llama-api", type=str, help="Set the address of server.cpp in llama.cpp(default: http://127.0.0.1:8080)", default='http://127.0.0.1:8080')
parser.add_argument("--api-key", type=str, help="Set the api key to allow only few user(default: NULL)", default=API_KEY)
parser.add_argument("--host", type=str, help="Set the ip address to listen.(default: 127.0.0.1)", default='127.0.0.1')
parser.add_argument("--port", type=int, help="Set the port to listen.(default: 8081)", default=8081)

args = parser.parse_args()


def is_present(json, key):
    try:
        buf = json[key]
    except KeyError:
        return False
    return True


#convert chat to prompt
def convert_chat(messages):
    stop = args.stop.replace("\\n", "\n")
    if len(messages) == 1:
        system_message_instruction = "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question."
        system_message_context = messages[0]["content"]
        prompt = f"""
        <s>[INST] <<SYS>>
        {system_message_instruction}
        <</SYS>>

        {system_message_context} [/INST]{stop}
        """.strip()

        return prompt, True
    else:
        system_message = [message["content"] for message in messages if message["role"]=="system"][0]
        system_message_instruction = "You are a helpful, respectful and professional assistant for Aznor. "
        system_message_instruction += system_message.split("\n----------------\n")[0]
        system_message_instruction += " ONLY answer questions that was asked by the user about Aznor, the schools he attended and companies he worked for. If there are no questions, thank the user and end the conversation."
        system_message_context = system_message.split("\n----------------\n")[1]
        question = [message["content"] for message in messages if message["role"]=="user"][0]
        question = question.replace(REPHRASED_TOKEN, "")
        prompt = f"""
        <s>[INST] <<SYS>>
        {system_message_instruction}
        <</SYS>>

        {system_message_context}
        Question: {question} [/INST]{stop}
        """.strip()

        return prompt, False


def make_postData(body, chat=False, stream=False):
    postData = {}
    if (chat):
        postData["prompt"], is_rephrase_request = convert_chat(body["messages"])
    else:
        postData["prompt"] = body["prompt"]
    if(is_present(body, "temperature")): postData["temperature"] = body["temperature"]
    if(is_present(body, "top_k")): postData["top_k"] = body["top_k"]
    if(is_present(body, "top_p")): postData["top_p"] = body["top_p"]
    if(is_present(body, "max_tokens")): postData["n_predict"] = body["max_tokens"]
    if(is_present(body, "presence_penalty")): postData["presence_penalty"] = body["presence_penalty"]
    if(is_present(body, "frequency_penalty")): postData["frequency_penalty"] = body["frequency_penalty"]
    if(is_present(body, "repeat_penalty")): postData["repeat_penalty"] = body["repeat_penalty"]
    if(is_present(body, "mirostat")): postData["mirostat"] = body["mirostat"]
    if(is_present(body, "mirostat_tau")): postData["mirostat_tau"] = body["mirostat_tau"]
    if(is_present(body, "mirostat_eta")): postData["mirostat_eta"] = body["mirostat_eta"]
    if(is_present(body, "seed")): postData["seed"] = body["seed"]
    if(is_present(body, "logit_bias")): postData["logit_bias"] = [[int(token), body["logit_bias"][token]] for token in body["logit_bias"].keys()]
    if (args.stop != ""):
        postData["stop"] = [args.stop]
    else:
        postData["stop"] = []
    if(is_present(body, "stop")): postData["stop"] += body["stop"]
    postData["n_keep"] = -1
    postData["stream"] = stream

    return postData, is_rephrase_request

def make_resData(data, chat=False, promptToken=[]):
    resData = {
        "id": "chatcmpl" if (chat) else "cmpl",
        "object": "chat.completion" if (chat) else "text_completion",
        "created": int(time.time()),
        "truncated": data["truncated"],
        "model": "LLaMA_CPP",
        "usage": {
            "prompt_tokens": data["tokens_evaluated"],
            "completion_tokens": data["tokens_predicted"],
            "total_tokens": data["tokens_evaluated"] + data["tokens_predicted"]
        }
    }
    if (len(promptToken) != 0):
        resData["promptToken"] = promptToken
    if (chat):
        #only one choice is supported
        resData["choices"] = [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": data["content"],
            },
            "finish_reason": "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"
        }]
    else:
        #only one choice is supported
        resData["choices"] = [{
            "text": data["content"],
            "index": 0,
            "logprobs": None,
            "finish_reason": "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"
        }]
    return resData

def make_resData_stream(data, chat=False, time_now = 0, start=False):
    resData = {
        "id": "chatcmpl" if (chat) else "cmpl",
        "object": "chat.completion.chunk" if (chat) else "text_completion.chunk",
        "created": time_now,
        "model": "LLaMA_CPP",
        "choices": [
            {
                "finish_reason": None,
                "index": 0
            }
        ]
    }
    if (chat):
        if (start):
            resData["choices"][0]["delta"] =  {
                "role": "assistant"
            }
        else:
            resData["choices"][0]["delta"] =  {
                "content": data["content"]
            }
            if (data["stop"]):
                resData["choices"][0]["finish_reason"] = "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"
    else:
        resData["choices"][0]["text"] = data["content"]
        if (data["stop"]):
            resData["choices"][0]["finish_reason"] = "stop" if (data["stopped_eos"] or data["stopped_word"]) else "length"

    return resData


@app.route('/chat/completions', methods=['POST'])
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    if (args.api_key != "" and request.headers["Authorization"].split()[1] != args.api_key):
        return Response(status=403)
    body = request.get_json()
    stream = False
    tokenize = False
    if(is_present(body, "stream")): stream = body["stream"]
    if(is_present(body, "tokenize")): tokenize = body["tokenize"]
    postData, is_rephrase_request = make_postData(body, chat=True, stream=stream)

    promptToken = []
    if (tokenize):
        tokenData = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/tokenize"), data=json.dumps({"content": postData["prompt"]})).json()
        promptToken = tokenData["tokens"]

    if (not stream):
        data = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/completion"), data=json.dumps(postData))
        print(data.json())
        resData = make_resData(data.json(), chat=True, promptToken=promptToken)
        return jsonify(resData)
    else:
        def generate():
            data = requests.request("POST", urllib.parse.urljoin(args.llama_api, "/completion"), data=json.dumps(postData), stream=True)
            time_now = int(time.time())
            resData = make_resData_stream({}, chat=True, time_now=time_now, start=True)
            yield 'data: {}\n'.format(json.dumps(resData))
            for line in data.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    resData = make_resData_stream(json.loads(decoded_line[6:]), chat=True, time_now=time_now)
                    if is_rephrase_request:
                        resData["choices"][0]["delta"]["content"] = REPHRASED_TOKEN + resData["choices"][0]["delta"]["content"]
                    # print(resData["choices"][0]["delta"]["content"])
                    yield 'data: {}\n'.format(json.dumps(resData))
        return Response(generate(), mimetype='text/event-stream')
    

if __name__ == '__main__':
    app.run(args.host, port=args.port)
