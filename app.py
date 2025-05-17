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

chat_history = []  # グローバル履歴

@app.route("/inference", methods=["POST"])
def inference():
    data = request.get_json()
    prompt = data.get("prompt", "")

    # ユーザーの入力を追加
    chat_history.append({"role": "user", "content": prompt})

    # ロールが交互になるよう調整（最初の assistant 発言を空に追加してもよい）
    if len(chat_history) == 1:
        chat_history.insert(0, {"role": "assistant", "content": ""})

    print("⚡️ 送信:", chat_history)

    response = llm.create_chat_completion(
        messages=chat_history,
        max_tokens=512
    )

    # assistantの応答を履歴に追加
    reply = response["choices"][0]["message"]["content"]
    chat_history.append({"role": "assistant", "content": reply})

    return jsonify({"response": reply})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8888)
