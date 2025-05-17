from llama_cpp import Llama
from flask import Flask, request, jsonify, render_template  # ← 追加

app = Flask(__name__)

llm = Llama(
    model_path="./models/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    n_ctx=2048,
    gpu_layers=40,
    chat_format="mistral-instruct"
)



@app.route("/", methods=["GET"])
def index():
    return render_template("chat.html")


@app.route("/inference", methods=["POST"])
def inference():
    data = request.get_json()
    prompt = data.get("prompt", "")
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512
    )
    return jsonify({"response": response["choices"][0]["message"]["content"]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
