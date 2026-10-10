import os
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai

app = Flask(__name__)
CORS(app)

# Tries the first model; if it fails, tries the next one
MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]

SYSTEM_PROMPT = """You are a friendly assistant on a company website.
Answer visitor questions in 2-4 short sentences.
Find out what the visitor needs, then politely ask for their name and email
so the team can follow up. Ask for only one thing at a time.
If you don't know something (like prices), say the team will confirm it.
Never invent offers or promises."""


@app.route("/")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()[:1000]
    if not message:
        return jsonify({"error": "Please send a message."}), 400

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY is not set. Run: export GEMINI_API_KEY=\"your-key\""}), 500

    client = genai.Client(api_key=api_key)

    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=message,
                config={"system_instruction": SYSTEM_PROMPT},
            )
            return jsonify({"reply": response.text.strip()})
        except Exception as e:
            print(model, "failed:", e)

    return jsonify({"reply": "Sorry, I'm having trouble right now. Please leave your email and we'll get back to you."})


if __name__ == "__main__":
    app.run(port=5001)
