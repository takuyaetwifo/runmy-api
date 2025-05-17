from flask import Flask, request, jsonify, render_template
import requests

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    payload = {
        "model": MODEL_NAME,
        "prompt": user_input,
        "stream": False
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload)
        reply = res.json()["response"]
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)

