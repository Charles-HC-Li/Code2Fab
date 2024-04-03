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
from openai import OpenAI  # Import OpenAI library

logging.basicConfig(level=logging.INFO)

# client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", 'Insert you api key here'))
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 3000))


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@app.route("/generate-stl", methods=["POST"])
def generate_stl():
    code = request.json.get("code")
    try:
        result = subprocess.run(
            ["openscad", "-o", "./static/models/temp.stl", "-"],
            input=code,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            print(f"exec error: {result.stderr}")
            return jsonify(error="Failed to generate STL file"), 500

        print(f"stdout: {result.stdout}")
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
        data = request.json
        text = data["text"]
        code = data["code"]
        logging.info(f"Received text: {text}")

        template = """
        {text}
        ***Report Begins***
        {code}
        ***Report Ends***
        Follow the template below to output the result:
        ***Template Begins***

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
                        {"role": "system", "content": "hello"},
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
        img_url = "data:image/jpeg;base64," + encode_image(image_path)

        template = f"""
        {text}
        ***Report Begins***
        {code}
        ***Report Ends***
        Follow the template below to output the result:
        ***Template Begins***

        ***Template Ends***
        """

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": template},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            }
        ]

        def gpt_action(inner_messages):
            try:
                response_stream = client.chat.completions.create(
                    model="gpt-4-vision-preview",
                    messages=inner_messages,
                    max_tokens=300,
                    stream=True,
                )
                for chunk in response_stream:
                    item = chunk.choices[0]
                    logging.info(f"Processing item: {item}")
                    delta = item.delta if item.delta else None
                    if delta and delta.content:
                        content = delta.content
                        logging.info(f"Item content: {content}")
                        yield content.encode("utf-8")
                    else:
                        continue
            except Exception as e:
                logging.error(f"Error processing GPT action: {e}")
                error_message = "Error: " + str(e)
                yield error_message.encode("utf-8")

        return Response(gpt_action(messages), mimetype="text/event-stream")

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/api/analysis", methods=["POST"])
def analysis():
    try:
        data = request.json
        text = data["text"]
        code = data["code"]
        logging.info(f"Received text: {text}")

        template = """
        {text}
        ***Report Begins***
        {code}
        ***Report Ends***
        Follow the template below to output the result:
        ***Template Begins***

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
                        {"role": "system", "content": "hello"},
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
        data = request.json
        text = data["text"]
        code = data["code"]
        logging.info(f"Received text: {text}")

        template = """
        {text}
        ***Report Begins***
        {code}
        ***Report Ends***
        Follow the template below to output the result:
        ***Template Begins***

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
                        {"role": "system", "content": "hello"},
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
