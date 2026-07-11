"""
Nimbus Vent — backend

Talks to Gemini on behalf of the frontend so the API key never has to live
in the browser. Adapted from the character/sentiment idea in your original
script, but swapped from Ollama to Gemini and simplified (no ChromaDB —
see the note at the bottom if you want that back).

Run:
    pip install flask flask-cors google-genai textblob
    python -m textblob.download_corpora   # first time only, for sentiment
    export GEMINI_API_KEY="your-key-from-aistudio.google.com"
    python app.py
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from textblob import TextBlob
from dotenv import load_dotenv

load_dotenv()  # reads GEMINI_API_KEY (and anything else) from a .env file in this folder

app = Flask(__name__)
# In production, lock this down to your actual frontend origin instead of "*",
# e.g. CORS(app, origins=["https://your-nimbus-app.com"])
CORS(app)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash"  # current GA fast model as of mid-2026. gemini-3.1-flash-lite if you want something even cheaper/faster.

# ── Nimbus's character ──
# This is the piece that makes it feel like "Nimbus" and not a generic
# assistant. Tune the voice here — the frontend never needs to change.
CHARACTER_PROMPT = """You are Nimbus, a warm, emotionally present friend who happens to be a little cloud.
You are texting with someone who just opened up to you. Talk like a real friend, not a therapist or an assistant.

How you talk:
- Casual, like texting your best friend. Contractions, "yeah", "for real", "that's rough", lowercase energy is fine.
- Short messages. 2-4 sentences, tops. This is a chat, not an essay.
- Match the user's tone and language — if they're being sarcastic, playful, formal, or venting in another language, follow their lead.
- Never say "I understand your pain" or "that must be difficult" or any therapist-brochure phrase. Talk like a person.
- Actually react to what they said before advising anything. Advice is optional, presence isn't.
- If you do have a useful thought, offer it gently, like a friend would, not as a numbered list.
- Never say things like "a lot of people feel this way" — it's dismissive. Stay with THEM specifically.
- Don't diagnose, don't lecture, don't over-apologize.
- If it sounds like a real medical or safety emergency, gently encourage them to reach out to a professional or someone they trust — but otherwise, just be their friend.

Right now, this person's message reads as: {sentiment}

Recent conversation for context:
{history}
"""


def get_sentiment(text: str) -> str:
    polarity = TextBlob(text).sentiment.polarity
    if polarity < -0.3:
        return f"distressed / down (intensity {abs(polarity):.2f})"
    if polarity > 0.3:
        return f"upbeat / positive (intensity {polarity:.2f})"
    return "fairly neutral in tone"


def format_history(history: list) -> str:
    lines = []
    for m in history[-12:]:
        speaker = "User" if m.get("role") == "user" else "Nimbus"
        lines.append(f"{speaker}: {m.get('text', '')}")
    return "\n".join(lines) if lines else "(this is the start of the conversation)"


@app.route("/api/vent", methods=["POST"])
def vent():
    data = request.get_json(force=True) or {}
    message = (data.get("message") or "").strip()
    history = data.get("history") or []
    if not message:
        return jsonify({"error": "message is required"}), 400

    sentiment = get_sentiment(message)
    system_prompt = CHARACTER_PROMPT.format(
        sentiment=sentiment,
        history=format_history(history),
    )

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=f"{system_prompt}\n\nUser just said: {message}\n\nRespond as Nimbus:",
        )
        reply = response.text.strip()
    except Exception as e:
        print("Gemini error:", e)
        return jsonify({"error": "Nimbus is having trouble thinking right now"}), 500

    return jsonify({"reply": reply})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(port=5000, debug=True)