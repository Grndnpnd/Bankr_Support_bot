# Bankr Discord Support Bot

A semantic search–powered Discord bot that answers Bankr platform questions using relevant documentation. Proactively reaches out to users who seem lost, responds when @mentioned, and maintains per-user conversation context.

---

## Features

- **Semantic doc search** — ChromaDB + `all-MiniLM-L6-v2` embeds Bankr docs and retrieves only relevant chunks per question
- **Multilingual support** — Detects English, Simplified Chinese (简体中文), and Korean (한국어); responds in the user's language
- **Proactive outreach** — Monitors channels for support intent patterns; offers help without being asked
- **Conversation memory** — Per-user conversation history with TTL; disengages when user says thanks / done
- **Ollama LLM backend** — Works with any Ollama cloud model; falls back gracefully on errors
- **Channel routing** — Directs users to `#bug-reports` for confirmed bugs, `#partnership-request` for biz inquiries

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/KodaTCG/Bankr-Discord-Support-Bot.git
cd Bankr-Discord-Support-Bot
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required variables in `.env`:

| Variable | Description |
|---|---|
| `DISCORD_TOKEN` | Your Discord bot token from the [Developer Portal](https://discord.com/developers/applications) |
| `OLLAMA_API_KEY` | Your Ollama API key (required for cloud models) |
| `OLLAMA_URL` | Ollama endpoint (default: `https://ollama.com`) |
| `OLLAMA_MODEL` | Model to use (default: `glm-5:cloud`) |
| `DOCS_URL` | URL to fetch docs from (default: `https://docs.bankr.bot/llms-full.txt`) |
| `DOCS_REFRESH_HOURS` | How often to re-fetch docs (default: `6`) |
| `MONITORED_CHANNEL_IDS` | Comma-separated Discord channel IDs to monitor (empty = all) |

### 3. Create the Discord bot

1. Go to [discord.com/developers/applications](https://discord.com/developers/applications)
2. Create a New Application → give it a name (e.g. "Bankr Support Bot")
3. Go to **Bot** → copy the **Token** into `DISCORD_TOKEN`
4. Enable these **Privileged Gateway Intents**:
   - ✅ Message Content Intent
   - ✅ Server Members Intent
5. Go to **OAuth2 → URL Generator**:
   - Check scopes: `bot`, `applications.commands`
   - Bot permissions: `Send Messages`, `Read Message History`, `Mention Everyone`
   - Use the generated URL to invite the bot to your server

### 4. Run it

```bash
python bot.py
```

For production, use `screen`/`tmux` or a systemd service to keep it running.

---

## Architecture

```
User message
    │
    ├─ Mention / reply to bot?  → handle_support_message()
    ├─ Active conversation?     → handle_support_message()
    └─ Passive monitoring       → detect_support_intent()
                                     │
                                     ├─ Intent detected → proactive_offer()
                                     └─ No intent       → ignore
    │
    v
detect_support_intent()   ← regex scoring against 30+ patterns
    │
    v
semantic_docs.query()    ← ChromaDB + MiniLM embedding
    │
    v
build_system_prompt()    ← inject relevant doc chunks
    │
    v
ollama.chat()            ← generate response
    │
    v
discord.reply()           ← send in-thread
```

---

## Key Design Decisions

- **No DB persistence** — conversation state is in-memory only; restarting the bot clears history. Intentionale: keeps ops simple for a support bot.
- **Single collection, reset on refresh** — docs are re-fetched and fully re-indexed every `DOCS_REFRESH_HOURS`. ChromaDB collection is wiped and rebuilt.
- **Temperature 0.3** — keeps responses focused and deterministic.
- **Chunk overlap 80 chars** — ensures context doesn't get split mid-sentence.

---

## Deploying to production

### Systemd service (recommended)

```ini
[Unit]
Description=Bankr Discord Support Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/bankr-support-bot
ExecStart=/usr/bin/python3 bot.py
Restart=always
RestartSec=10
EnvironmentFile=/home/ubuntu/bankr-support-bot/.env

[Install]
WantedBy=multi-user.target
```

```bash
sudo cp bankr-support-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable bankr-support-bot
sudo systemctl start bankr-support-bot
```

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY bot.py .
COPY .env.example .env
CMD ["python", "bot.py"]
```

```bash
docker build -t bankr-support-bot .
docker run --env-file .env bankr-support-bot
```

---

## License

MIT
