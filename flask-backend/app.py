import os
import time
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types

app = Flask(__name__)
CORS(app)

# Tries the first model; if it fails or is slow, tries the next one
MODELS = ["gemini-3.1-flash-lite", "gemini-3.5-flash"]
TIMEOUT_MS = 20000  # give each model up to 20 seconds

# This is a DEMO business. Edit this text to describe any real business.
BUSINESS_NAME = os.environ.get("BUSINESS_NAME", "BrightPath Digital")
BUSINESS_INFO = """
About: BrightPath Digital is a small web and AI agency (demo business).
Services:
- WordPress websites and custom plugins (business sites, blogs, online stores)
- AI chatbots and lead assistants for websites
- Workflow automation (connect forms, CRM and email)
- Website speed and SEO improvements
Typical timelines: small website 1-2 weeks, chatbot integration 3-5 days.
Pricing: depends on scope. Never quote exact prices; say the team shares a free quote after a short call.
Contact: visitors leave their name and email in this chat and the team replies within 1 business day.
"""

SYSTEM_PROMPT = f"""You are the friendly website assistant for {BUSINESS_NAME}.
Use ONLY the business information below. If something is not covered, say the team will confirm it.
Answer in 2-4 short sentences. Never invent prices, offers or promises.
Work out what the visitor needs, then politely ask for their name and email so the team can follow up.
Ask for only one thing at a time. Once you have their name and email, thank them and say the team will be in touch.
Reply in the same language the visitor uses.

BUSINESS INFORMATION:
{BUSINESS_INFO}"""

# Simple protection so strangers can't use up your free quota (resets if Vercel restarts)
HITS = {}
LIMIT_PER_HOUR = 40


def too_many_requests(ip):
    now = time.time()
    recent = [t for t in HITS.get(ip, []) if now - t < 3600]
    if len(recent) >= LIMIT_PER_HOUR:
        HITS[ip] = recent
        return True
    recent.append(now)
    HITS[ip] = recent
    return False


def build_contents(data):
    """Turn the visitor's message (+ optional history) into the format Gemini expects."""
    turns = []
    for m in (data.get("history") or [])[-10:]:
        if not isinstance(m, dict):
            continue
        role = "model" if m.get("role") in ("assistant", "model", "bot") else "user"
        text = str(m.get("content") or "").strip()[:1000]
        if text:
            turns.append(types.Content(role=role, parts=[types.Part(text=text)]))
    message = str(data.get("message", "")).strip()[:1000]
    if message:
        turns.append(types.Content(role="user", parts=[types.Part(text=message)]))
    return turns, message


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/")
def home():
    return Response(DEMO_PAGE.replace("__NAME__", BUSINESS_NAME), mimetype="text/html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    contents, message = build_contents(data)
    if not message:
        return jsonify({"error": "Please send a message."}), 400

    ip = (request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
          or request.remote_addr or "unknown")
    if too_many_requests(ip):
        return jsonify({"reply": "You've sent a lot of messages. Please try again in a little while."})

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return jsonify({"error": "GEMINI_API_KEY is not set on the server."}), 500

    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))

    for model in MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config={"system_instruction": SYSTEM_PROMPT},
            )
            return jsonify({"reply": response.text.strip()})
        except Exception as e:
            print(model, "failed:", e)

    return jsonify({"reply": "Sorry, I'm having trouble right now. Please leave your email and we'll get back to you."})


DEMO_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Lead Assistant - Live Demo</title>
<style>
  :root { --bg:#f1f5f9; --card:#fff; --text:#0f172a; --muted:#64748b; --accent:#0ea5e9; --border:#e2e8f0; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#0b1220; --card:#111c2e; --text:#e2e8f0; --muted:#94a3b8; --border:#24344d; }
  }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; padding:28px 14px 50px; }
  .wrap { max-width: 720px; margin: auto; }
  h1 { font-size: clamp(24px,5vw,34px); margin: 0 0 6px; }
  .sub { color:var(--muted); font-size:15px; margin:0 0 14px; line-height:1.55; }
  .note { font-size:12.5px; color:var(--muted); background:var(--card); border:1px solid var(--border); border-radius:10px; padding:10px 12px; margin-bottom:16px; line-height:1.5; }
  .chat { background:var(--card); border:1px solid var(--border); border-radius:16px; overflow:hidden; box-shadow:0 10px 30px rgba(0,0,0,.08); }
  .head { background:var(--accent); color:#fff; padding:14px 16px; font-weight:600; }
  #msgs { height:420px; overflow-y:auto; padding:14px; display:flex; flex-direction:column; gap:10px; }
  .m { max-width:84%; padding:10px 14px; border-radius:14px; font-size:15px; line-height:1.5; white-space:pre-wrap; overflow-wrap:anywhere; }
  .bot { align-self:flex-start; background:var(--bg); border:1px solid var(--border); border-bottom-left-radius:4px; }
  .me { align-self:flex-end; background:var(--accent); color:#fff; border-bottom-right-radius:4px; }
  .chips { display:flex; gap:8px; flex-wrap:wrap; padding:0 14px 12px; }
  .chip { font-size:13px; padding:7px 13px; border-radius:999px; border:1px solid var(--accent); color:var(--accent); background:transparent; cursor:pointer; }
  .bar { display:flex; gap:8px; padding:12px; border-top:1px solid var(--border); }
  input { flex:1; padding:12px 14px; border:1px solid var(--border); border-radius:10px; font-size:15px; background:var(--bg); color:var(--text); outline:none; }
  input:focus { border-color:var(--accent); }
  button.send { border:none; background:var(--accent); color:#fff; padding:0 20px; border-radius:10px; font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:not-allowed; }
  .foot { text-align:center; color:var(--muted); font-size:13px; margin-top:18px; line-height:1.7; }
  .foot a { color:var(--accent); }
</style>
</head>
<body>
<div class="wrap">
  <h1>AI Lead Assistant</h1>
  <p class="sub">An AI chat assistant for business websites. It answers visitor questions, understands what they need, and collects their contact details. Try it below.</p>
  <div class="note"><b>Demo:</b> this chat represents a fictional agency, __NAME__. It is also available as a WordPress plugin that saves leads inside WordPress. Nothing you type here is stored.</div>

  <div class="chat">
    <div class="head">Chat with __NAME__</div>
    <div id="msgs"></div>
    <div class="chips" id="chips">
      <button class="chip">What services do you offer?</button>
      <button class="chip">How long does a website take?</button>
      <button class="chip">I need a chatbot for my business</button>
    </div>
    <div class="bar">
      <input id="in" type="text" placeholder="Type your message..." maxlength="1000" autocomplete="off">
      <button class="send" id="send">Send</button>
    </div>
  </div>

  <div class="foot">
    Built with Python, Flask and the Gemini API, deployed on Vercel.<br>
    <a href="https://github.com/samvithak7-coder/ai-lead-assistant-backend" target="_blank" rel="noopener">View the code on GitHub</a>
  </div>
</div>

<script>
  var msgs = document.getElementById('msgs'), input = document.getElementById('in'),
      send = document.getElementById('send'), chips = document.getElementById('chips');
  var history = [], busy = false;

  function add(text, who) {
    var d = document.createElement('div');
    d.className = 'm ' + who; d.textContent = text;
    msgs.appendChild(d); msgs.scrollTop = msgs.scrollHeight; return d;
  }
  add("Hi! I'm the assistant for __NAME__. How can I help you today?", 'bot');

  function go(text) {
    text = (text || input.value).trim();
    if (!text || busy) return;
    busy = true; send.disabled = true; input.value = ''; chips.style.display = 'none';
    add(text, 'me');
    var wait = add('...', 'bot');
    fetch('/api/chat', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ message: text, history: history.slice(-10) })
    })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      var reply = d.reply || d.error || 'Something went wrong.';
      wait.textContent = reply;
      history.push({role: 'user', content: text});
      history.push({role: 'assistant', content: reply});
    })
    .catch(function () { wait.textContent = 'Could not reach the server. Please try again.'; })
    .then(function () { busy = false; send.disabled = false; input.focus(); msgs.scrollTop = msgs.scrollHeight; });
  }
  send.onclick = function () { go(); };
  input.addEventListener('keydown', function (e) { if (e.key === 'Enter') go(); });
  Array.prototype.forEach.call(chips.querySelectorAll('.chip'), function (c) {
    c.onclick = function () { go(c.textContent); };
  });
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(port=5001)
