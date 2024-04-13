"""
This code defines a Flask application that serves as an API for generating STL files, saving and loading OpenSCAD code files, and performing model description and code matching tasks using OpenAI's GPT-4 models.

The API endpoints include:
- /generate-stl: Accepts a POST request with OpenSCAD code and generates an STL file.
- /save-code: Accepts a POST request with OpenSCAD code and saves it as a file.
- /get-files: Returns a list of saved code files.
- /load-code/<file_name>: Returns the content of a specific code file.
- /api/describe: Accepts a POST request with text, code, and an image, and uses GPT-4 to generate a description of the 3D model.
- /api/match: Accepts a POST request with text, code, and an image, and uses GPT-4 to match the different parts of the 3D model to the corresponding code.
- /api/analysis: Accepts a POST request with text and code, and uses GPT-4 to analyze the OpenSCAD code.
- /api/improve: Accepts a POST request with text and code, and uses GPT-4 to provide suggestions for improving the code.
The code also includes configuration settings for the GPT-4 models, agent definitions for model descriptor, code interpreter, and user proxy, and a Flask route for serving the index.html file.

Note: The code includes sensitive information such as API keys and authorization headers. Make sure to handle this information securely in a production environment.
"""
from flask import (
    Flask,
    request,
    send_from_directory,
    jsonify,
    Response,
    stream_with_context,
)
import base64
import requests
import logging
import os
import subprocess
from openai import OpenAI 
import autogen
from autogen import Agent, AssistantAgent, ConversableAgent, UserProxyAgent
import requests
from autogen.agentchat.contrib.multimodal_conversable_agent import (
    MultimodalConversableAgent,
)

logging.basicConfig(level=logging.INFO)

api_key = ""
# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", 'Insert you api key here'))
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))

"""AutoGen Config"""
config_llm_4v = [{"model": "gpt-4-vision-preview", "api_key": api_key}]

config_llm_4 = [{"model": "gpt-4-1106-preview", "api_key": api_key}]

config_list_4v = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": config_llm_4v,
    "temperature": 0,
    "max_tokens": 2000,
}

config_list_4 = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": config_llm_4,
    "temperature": 0,
}

model_descriptor = MultimodalConversableAgent(
    name="3D_model_descriptor",
    max_consecutive_auto_reply=10,
    llm_config=config_list_4v,
    system_message="""
As a good 3D model descriptor, you will receive images from the OpenSCAD 3D model and generate a detailed description of the 3D model, describing what the 3D model is and what parts it consists of. After that, you will work with the code interpreter to match the different parts of the model to the code that generates this corresponding part.
Use the following format for output:
***Report Begins***
##Description of the model##
[Insert the description of the model here, highlighting key elements.]

##Summary of the model##
[Insert the summary of the model here, contains all the components.]
***Report Ends***
""",
)

code_interpreter = autogen.AssistantAgent(
    name="code_interpreter",
    llm_config=config_list_4,
    system_message="""
As an expert in OpenSCAD code interpretation, you will receive a set of OpenSCAD code. For a given piece of code, you will work with the 3D model descriptor to connect the different parts of the 3d model and their corresponding code.
Use the following format for output:
***Report Begins***
##Codes##
"Code1", [The corresponding part in the model], 
[content of Code1]
"Code2", [The corresponding part in the model], 
[content of Code1]
...
***Report Ends***
""",
)

user_proxy = autogen.UserProxyAgent(
    name="User_proxy",
    system_message="A human admin.",
    human_input_mode="NEVER",  # Try between ALWAYS or NEVER
    max_consecutive_auto_reply=0,
    code_execution_config={
        "use_docker": False
    },  # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)

groupchat = autogen.GroupChat(
    agents=[model_descriptor, code_interpreter,
            user_proxy], messages=[], max_round=5
)
manager = autogen.GroupChatManager(
    groupchat=groupchat, llm_config=config_list_4)

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 3000))


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def upload_image(image_path):
    headers = {'Authorization': 'ICcEBQDFvmdJGPfwpGMSYxgkSEYHnVyw'}
    url = "https://sm.ms/api/v2/upload"
    files = {"smfile": open(image_path, "rb")}
    response = requests.post(url, files=files, headers = headers)
    if response.status_code == 200:
        data = response.json()
        if data["code"] == "success":
            return data["data"]["url"]
    return None

current_process = None
@app.route("/generate-stl", methods=["POST"])
def generate_stl():
    global current_process
    code = request.json.get("code")
    if current_process and current_process.poll() is None:
        current_process.terminate()
    try:
        current_process = subprocess.Popen(
            ["openscad", "-o", "./static/models/temp.stl", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = current_process.communicate(input=code)
        if current_process.returncode != 0:
            print(f"exec error: {stderr}")
            return jsonify(error="Failed to generate STL file"), 500
        print(f"stdout: {stdout}")
        return jsonify(stlUrl="/static/models/temp.stl")
    except Exception as e:
        print(f"Execution error: {e}")
        return jsonify(error="Failed to generate model."), 500


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


CODE_DIRECTORY = "./static/code"
unnamed_counter = 1


@app.route("/save-code", methods=["POST"])
def save_code():
    global unnamed_counter
    data = request.json
    code = data["code"]
    file_name = data.get("fileName")

    if not file_name:
        file_name = f"unnamed_version{unnamed_counter}.scad"
        unnamed_counter += 1

    file_path = os.path.join(CODE_DIRECTORY, file_name)

    os.makedirs(CODE_DIRECTORY, exist_ok=True)

    try:
        with open(file_path, "w") as file:
            file.write(code)
        return "File saved"
    except Exception as e:
        print(f"Error saving file: {e}")
        return "Error saving file", 500


@app.route("/get-files", methods=["GET"])
def get_files():
    try:
        files = os.listdir(CODE_DIRECTORY)
        return jsonify(files)
    except Exception as e:
        print(f"Error reading directory: {e}")
        return "Error reading directory", 500


@app.route("/load-code/<file_name>", methods=["GET"])
def load_code(file_name):
    file_path = os.path.join(CODE_DIRECTORY, file_name)
    try:
        with open(file_path, "r") as file:
            code = file.read()
        return jsonify({"code": code})
    except Exception as e:
        print(f"Error loading file: {file_name}, {e}")
        return "Error loading file", 500


@app.route("/api/describe", methods=["POST"])
def describe():
    try:
        text = request.form["text"]
        code = request.form["code"]
        image = request.files["image"]
        logging.info(f"Received text: {text}")

        # Save the image and encode it
        image_path = "./static/img/temp.jpg"
        image.save(image_path)
        img_url = "data:image/jpeg;base64," + encode_image(image_path)

        template = """
        {text}
        ***Report Begins***
        {code}
        ***Report Ends***
        Follow the template below to output the result:
        ***Template Begins***

        ***Template Ends***
        """

        def gpt_action(img_url):
            try:
                completion = client.chat.completions.create(
                    model="gpt-4-turbo",
                    temperature=0.0,
                    timeout=10,
                    stream=True,
                    messages=[
                        {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Given the same 3D model viewed from different angles, describe the shape such that a blind user could understand it."},
                            {
                            "type": "image_url",
                            "image_url": {
                                "url": img_url,
                            },
                            },
                        ],
                        }
                    ],
                )
                for chunk in completion:
                    if chunk.choices[0].delta:
                        yield chunk.choices[0].delta.content.encode("utf-8")
                    else:
                        yield b"Processing...\n"
            except (AttributeError, TypeError) as e:
                if str(e) != "'NoneType' object has no attribute 'encode'":
                    yield "Error: " + str(e)

        return Response(gpt_action(img_url), mimetype="text/event-stream")

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/match", methods=["POST"])
def match():
    try:
        text = request.form["text"]
        code = request.form["code"]
        image = request.files["image"]
        logging.info(f"Received text: {text}")

        # Save the image and encode it
        image_path = "./static/img/temp.jpg"
        image.save(image_path)
        img_url = upload_image(image_path)

        template = f"""
You will first ask the model descriptor to describe what the model consists of and what it looks like, and then the code interpreter will work with the model descriptor to match the different parts of the model to the corresponding code:
<img {img_url}>.

Extra requirement: [{text}]
***Code Begins***
'''openscad'''
{code}
'''openscad'''
***Code Ends***

Use the follow format for output:
***Report Begins***
##Direct Reply for user's input##
[Insert the direct reply for user's input. DO NOT return this section if there isn't any extra requirement.]

##Description of the model##
[Insert the description of the model here, highlighting key elements.]

##Summary of the model##
[Insert the summary of the model here, contains all the components.]

##Codes##
"Code1", [The corresponding part in the model], 
[content of Code1]
"Code2", [The corresponding part in the model], 
[content of Code1]
...
***Report Ends***
"""
        user_proxy.initiate_chat(manager, message=template)

        message = code_interpreter.last_message()['content']

        try:
            return message
        except Exception as e:
            logging.error(f"Error processing GPT action: {e}")
            error_message = "Error: " + str(e)
            return error_message

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/analysis", methods=["POST"])
def analysis():
    try:
        text = request.form["text"]
        code = request.form["code"]
        image = request.files["image"]
        logging.info(f"Received text: {text}")

        template = """
As a code interpreter, you will receive a set of OpenSCAD code and analyze the code for a blind user to understand.
Given the Openscad code, you will analyze the code and provide a detailed description of the code, highlighting the key elements and code structure. After that, you will evaluate the code, highlighting the strengths and weaknesses. 
***Code Begins***
'''openscad'''
{code}
'''openscad'''
***Code Ends***

Use the follow format for output:
***Report Begins***

##Description of the openscad code##
[Insert the description of the openscad here, highlighting key elements and code structure.]

##Summary of the code##
[Insert the summary of the code here, contains all the components.]

##Evaluation of the code##
[Insert the evaluation of the code here, highlighting the strengths and weaknesses.]

##Codes##
"Code1", [Function of the code],[Suggestions for improvement] 
[content of Code1]
"Code2", [Function of the code],[Suggestions for improvement]
[content of Code1]
...
***Report Ends***
        """

        def gpt_action(text, code):
            try:
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    timeout=10,
                    stream=True,
                    messages=[
                        {
                            "role": "user",
                            "content": template.format(text=text, code=code),
                        },
                    ],
                )
                for chunk in completion:
                    if chunk.choices[0].delta:
                        yield chunk.choices[0].delta.content.encode("utf-8")
                    else:
                        yield b"Processing...\n"
            except (AttributeError, TypeError) as e:
                if str(e) != "'NoneType' object has no attribute 'encode'":
                    yield "Error: " + str(e)

        return Response(gpt_action(text, code), mimetype="text/event-stream")

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/improve", methods=["POST"])
def improve():
    try:
        text = request.form["text"]
        code = request.form["code"]
        image = request.files["image"]
        logging.info(f"Received text: {text}")

        template = """
As a professional code reviewer, you will receive a set of OpenSCAD code and provide suggestions for improving the code for a blind user to improve the code.
{text}
***Code Begins***
'''openscad'''
{code}
'''openscad'''
***Code Ends***

Follow the template below to output the result:
***Template Begins***
##Suggestions for improving the code##
[Insert the suggestions for improving the code here, highlighting key elements and code structure.]

##Evaluation of the code##
[Insert the evaluation of the code here, highlighting the strengths and weaknesses.]

##Details for Codes' improvement##
"Code1", [Function of the code],[Suggestions for improvement]
Original Code: [content of Code1]
Improved Code: [Improved content of Code1]

"Code2", [Function of the code],[Suggestions for improvement]
Original Code: [content of Code2]
Improved Code: [Improved content of Code2]
...
***Template Ends***
"""

        def gpt_action(text, code):
            try:
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    temperature=0.0,
                    timeout=10,
                    stream=True,
                    messages=[
                        {
                            "role": "user",
                            "content": template.format(text=text, code=code),
                        },
                    ],
                )
                for chunk in completion:
                    if chunk.choices[0].delta:
                        yield chunk.choices[0].delta.content.encode("utf-8")
                    else:
                        yield b"Processing...\n"
            except (AttributeError, TypeError) as e:
                if str(e) != "'NoneType' object has no attribute 'encode'":
                    yield "Error: " + str(e)

        return Response(gpt_action(text, code), mimetype="text/event-stream")

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
