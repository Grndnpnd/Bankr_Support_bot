"""
Bankr Support Bot
-----------------
- Monitors messages for support intent using keyword/pattern matching
- Proactively reaches out to users who seem to need help
- Responds when directly @mentioned
- Maintains per-user conversation history
- Uses semantic search (ChromaDB + MiniLM) to inject only relevant doc chunks
"""

import discord
import asyncio
import aiohttp
import re
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv
import os

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

DISCORD_TOKEN   = os.getenv("DISCORD_TOKEN")
OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY")
OLLAMA_URL      = os.getenv("OLLAMA_URL", "https://ollama.com")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "nemotron-3-nano:30b-cloud")

DOCS_URL              = os.getenv("DOCS_URL", "https://docs.bankr.bot/llms-full.txt")
DOCS_REFRESH_HOURS    = int(os.getenv("DOCS_REFRESH_HOURS", "6"))

# Channels to monitor (empty = all channels)
MONITORED_CHANNEL_IDS = [int(x) for x in os.getenv("MONITORED_CHANNEL_IDS", "").split(",") if x.strip()]

CONVERSATION_TTL_MINUTES = int(os.getenv("CONVERSATION_TTL_MINUTES", "30"))
REFLAG_COOLDOWN_MINUTES  = int(os.getenv("REFLAG_COOLDOWN_MINUTES", "15"))

# Semantic search tuning
CHUNK_SIZE          = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP       = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K_CHUNKS        = int(os.getenv("TOP_K_CHUNKS", "6"))
MAX_RETRIEVED_CHARS = int(os.getenv("MAX_RETRIEVED_CHARS", "8000"))

# ─── Intent Detection ─────────────────────────────────────────────────────────

SUPPORT_PATTERNS = [
    r"\bhow (do|can|to|does|would|should|come)\b",
    r"\bwhy (is|does|isn'?t|doesn'?t|won'?t|can'?t|would|did|didn'?t)\b",
    r"\bwhat (is|are|does|the|if|about|happens?|should)\b",
    r"\bwhere (do|can|is|are|should|would)\b",
    r"\bwhen (do|can|will|does|should|would)\b",
    r"\bcan (i|you|we|someone|it)\b",
    r"\b(not working|doesn'?t work|won'?t work|broke|broken|failed|failing|never works?)\b",
    r"\b(errors?|issues?|bugs?|problems?|glitch(?:es)?|crash(?:es)?)\b",
    r"\b(help|stuck|confused|unsure|unclear|lost|struggling)\b",
    r"\b(can'?t|cannot|couldn'?t|won'?t|doesn'?t|didn'?t|isn'?t|wasn'?t)\b",
    r"\b(trying to|tried to|attempting to|keeps? (failing|erroring|breaking))\b",
    r"\b(no idea|don'?t understand|don'?t know|not sure|idk|no clue)\b",
    r"\b(swaps?|swapping|traded?|trades?|trading|bankr|bnkr|bot)\b",
    r"\b(tokens?|launches?|launching|deploys?|deploying|deployed)\b",
    r"\b(wallets?|balances?|fees?|claiming|claims?|skills?)\b",
    r"\b(openclaw|api keys?|llm gateway|llm|agent)\b",
    r"\b(solana|base|ethereum|polygon|unichain)\b",
    r"\b(ugh|argh|wtf|wth|omg|frustrat\w*|annoying|annoyed|pissed|fuck this|scammers)\b",
    r"[?]{2,}",
    r"\b(wrong with|is wrong|went wrong|going wrong)\b",
    # Simplified Chinese
    r"[怎如何为什么什么哪][么样能会][办做用是去]?",
    r"[能可]以?[吗嘛]",
    r"[不没][能行知道会][用]?",
    r"[帮请].*[我忙助]",
    r"[错误问题故障][了吗]?",
    r"[失败无法不行][了吗]?",
    r"[交换兑换]",
    r"[钱包余额费用]",
    r"[代币发行部署]",
    r"[链上交易买卖]",
    r"[帮助支持问题]",
    # Korean
    r"어떻게|왜|무엇|뭐|어디|언제|어떤",
    r"할 수 있나요?|할 수 없|안 되|안되",
    r"모르겠|모르|헷갈|이해가 안",
    r"도와|도움|help",
    r"오류|에러|문제|버그|고장|실패",
    r"안 됩니다|작동이 안|작동 안|실행이 안",
    r"스왑|교환|거래|매수|매도",
    r"지갑|잔액|수수료|클레임",
    r"토큰|발행|배포|런치",
    r"체인|솔라나|이더리움|폴리곤|베이스",
]

COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in SUPPORT_PATTERNS]

MIN_MESSAGE_LENGTH = 10
INTENT_THRESHOLD   = 2

DISENGAGE_COMMANDS = {"!done", "!close", "!stop", "!bye", "!thanks", "!thank you"}

DISENGAGE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\b(thanks?|thank you|cheers|got it|perfect|solved|worked|fixed|all good|sorted)\b",
        r"\b(bye|goodbye|cya|see ya|later)\b",
        r"\bthat('?s| is) (all|it|enough|perfect|great|helpful)\b",
        r"\bno (more )?questions?\b",
        r"\bi'?m good( now)?\b",
        r"\b(nevermind|never mind|nvm|nm)\b",
        r"\b(works? now|working now|figured it out|got it working)\b",
        r"谢谢|感谢|没问题|解决了|好的|明白了|懂了|再见",
        r"감사합니다|감사해요|고마워|해결됐|됐어요|알겠습니다|알겠어요|괜찮아|안녕",
    ]
]


def detect_support_intent(message: str) -> tuple[bool, int]:
    if len(message) < MIN_MESSAGE_LENGTH:
        return False, 0
    score = sum(1 for p in COMPILED_PATTERNS if p.search(message))
    return score >= INTENT_THRESHOLD, score


# ─── Semantic Docs Manager ────────────────────────────────────────────────────

import chromadb
from chromadb.utils import embedding_functions


class SemanticDocsManager:
    def __init__(self):
        self.raw_content: str = ""
        self.last_fetched: datetime | None = None
        self._ready: bool = False

        self._client = chromadb.Client()
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self._collection = self._client.get_or_create_collection(
            name="bankr_docs",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    async def ensure_ready(self):
        now = datetime.utcnow()
        if (
            not self._ready
            or not self.last_fetched
            or now - self.last_fetched > timedelta(hours=DOCS_REFRESH_HOURS)
        ):
            await self._fetch_and_index()

    async def query(self, question: str) -> str:
        await self.ensure_ready()

        if not self._ready:
            return "Documentation unavailable. Please check docs.bankr.bot directly."

        results = self._collection.query(
            query_texts=[question],
            n_results=min(TOP_K_CHUNKS, self._collection.count()),
        )

        chunks = results["documents"][0] if results["documents"] else []
        if not chunks:
            return "No relevant documentation found."

        combined = "\n\n---\n\n".join(chunks)
        if len(combined) > MAX_RETRIEVED_CHARS:
            combined = combined[:MAX_RETRIEVED_CHARS] + "\n\n[Context trimmed — full docs at docs.bankr.bot]"

        return combined

    def _chunk_text(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n{2,}", text)
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current) + len(para) + 2 <= CHUNK_SIZE:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                overlap_text = current[-CHUNK_OVERLAP:] if current else ""
                current = (overlap_text + "\n\n" + para).strip() if overlap_text else para

                while len(current) > CHUNK_SIZE:
                    chunks.append(current[:CHUNK_SIZE])
                    current = current[CHUNK_SIZE - CHUNK_OVERLAP:]

        if current:
            chunks.append(current)

        return [c for c in chunks if len(c) > 40]

    def _index_docs(self):
        chunks = self._chunk_text(self.raw_content)
        log.info(f"Chunked into {len(chunks)} segments — embedding...")

        try:
            self._collection.delete(where={"source": "bankr_docs"})
        except Exception:
            pass  # Collection may be empty on first run

        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self._collection.add(
                documents=batch,
                ids=[f"chunk_{i + j}" for j in range(len(batch))],
                metadatas=[{"source": "bankr_docs", "index": i + j} for j in range(len(batch))],
            )

        log.info(f"Indexed {len(chunks)} chunks into ChromaDB")

    async def _fetch_and_index(self):
        log.info(f"Fetching docs from {DOCS_URL}...")
        try:
            headers = {"Accept-Encoding": "gzip, deflate"}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    DOCS_URL, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        self.raw_content = await resp.text()
                        self.last_fetched = datetime.utcnow()
                        log.info(f"Docs fetched ({len(self.raw_content):,} chars) — indexing...")
                        await asyncio.get_event_loop().run_in_executor(None, self._index_docs)
                        self._ready = True
                        log.info("Docs ready.")
                    else:
                        log.error(f"Failed to fetch docs: HTTP {resp.status}")
        except Exception as e:
            log.error(f"Error fetching/indexing docs: {e}")
            if not self.raw_content:
                self.raw_content = "Documentation unavailable."


# ─── Conversation Manager ─────────────────────────────────────────────────────

class ConversationManager:
    def __init__(self):
        self.conversations: dict = defaultdict(
            lambda: {"history": [], "last_active": datetime.utcnow()}
        )

    def _key(self, channel_id: int, user_id: int) -> tuple:
        return (channel_id, user_id)

    def add_message(self, channel_id: int, user_id: int, role: str, content: str):
        key = self._key(channel_id, user_id)
        self.conversations[key]["history"].append({"role": role, "content": content})
        self.conversations[key]["last_active"] = datetime.utcnow()
        if len(self.conversations[key]["history"]) > 20:
            self.conversations[key]["history"] = self.conversations[key]["history"][-20:]

    def get_history(self, channel_id: int, user_id: int) -> list:
        return self.conversations[self._key(channel_id, user_id)]["history"]

    def clear(self, channel_id: int, user_id: int):
        key = self._key(channel_id, user_id)
        if key in self.conversations:
            del self.conversations[key]

    def has_active_conversation(self, channel_id: int, user_id: int) -> bool:
        key = self._key(channel_id, user_id)
        if key not in self.conversations or not self.conversations[key]["history"]:
            return False
        return datetime.utcnow() - self.conversations[key]["last_active"] < timedelta(
            minutes=CONVERSATION_TTL_MINUTES
        )

    def cleanup_expired(self):
        now = datetime.utcnow()
        expired = [
            k for k, v in self.conversations.items()
            if now - v["last_active"] > timedelta(minutes=CONVERSATION_TTL_MINUTES)
        ]
        for k in expired:
            del self.conversations[k]
        if expired:
            log.info(f"Cleaned up {len(expired)} expired conversations")


# ─── Ollama Client ────────────────────────────────────────────────────────────

class OllamaClient:
    def __init__(self, base_url: str, model: str, api_key: str = None):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.api_key  = api_key

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def chat(self, messages: list[dict], system: str = "") -> str:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": all_messages,
            "stream": False,
            "options": {"temperature": 0.3},
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["message"]["content"]
                    else:
                        text = await resp.text()
                        log.error(f"Ollama error {resp.status}: {text}")
                        return "Sorry, I ran into an issue generating a response. Please try again."
        except asyncio.TimeoutError:
            return "Sorry, the response took too long. Please try again."
        except Exception as e:
            log.error(f"Ollama request failed: {e}")
            return "Sorry, I couldn't connect to the AI backend. Please try again later."


# ─── Bot ──────────────────────────────────────────────────────────────────────

class BankrSupportBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        super().__init__(intents=intents)

        self.docs          = SemanticDocsManager()
        self.conversations = ConversationManager()
        self.ollama        = OllamaClient(OLLAMA_URL, OLLAMA_MODEL, api_key=OLLAMA_API_KEY)

        self.recently_flagged:    dict[tuple, datetime] = {}
        self._handled_message_ids: set[int]             = set()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def on_ready(self):
        log.info(f"Logged in as {self.user} (ID: {self.user.id})")
        log.info(
            f"Monitoring {'all channels' if not MONITORED_CHANNEL_IDS else f'channels: {MONITORED_CHANNEL_IDS}'}"
        )
        log.info(
            f"Using model: {OLLAMA_MODEL} @ {OLLAMA_URL} "
            f"({'cloud API' if OLLAMA_API_KEY else 'no API key — local mode'})"
        )
        await self.docs.ensure_ready()
        asyncio.ensure_future(self._cleanup_loop())

    async def _cleanup_loop(self):
        while True:
            await asyncio.sleep(300)
            self.conversations.cleanup_expired()

            if len(self._handled_message_ids) > 1000:
                self._handled_message_ids = set(sorted(self._handled_message_ids)[-500:])

            now = datetime.utcnow()
            self.recently_flagged = {
                k: v for k, v in self.recently_flagged.items()
                if now - v < timedelta(minutes=REFLAG_COOLDOWN_MINUTES)
            }

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_system_prompt(self, relevant_docs: str) -> str:
        return f"""You are a helpful support bot for Bankr — a platform for AI agents that fund themselves through DeFi and token launching.

Your job is to help users with questions about Bankr using the relevant documentation excerpts below.

Guidelines:
- Answer based on the documentation provided. Be friendly, concise, and clear — this is Discord, not a formal ticket.
- If you show code or commands, use Discord markdown (wrap in backticks).
- Don't make up API endpoints, prices, or features that aren't in the docs.
- Keep responses short — if something needs a long explanation, break it into steps.
- Never repeat the user's question back to them. Just answer it directly.
- IMPORTANT: Detect the language of the user's message and always respond in the same language. If the user writes in Simplified Chinese (简体中文), respond entirely in Simplified Chinese. If the user writes in Korean (한국어), respond entirely in Korean. If they write in English, respond in English.

Channel routing — only use these as a last resort:
- If the user describes a specific bug, error message, or something actively broken that you cannot diagnose from the docs: acknowledge you can't resolve it and direct them to open a ticket in #bug-reports. Tell them to carefully read the instructions in that channel and that staff will be with them soon.
- If the user asks about partnerships, collaborations, or business inquiries: direct them to #partnership-request.
- If the user asks how to contact the team for general support (not a bug, not a partnership): let them know the best way is to ask you directly, and you'll escalate to #bug-reports only if needed.
- Do NOT send someone to #bug-reports just because a topic is complex or you're uncertain. Always attempt a best-effort answer from the docs first. Only route when you genuinely cannot help.

--- RELEVANT BANKR DOCUMENTATION ---
{relevant_docs}
--- END DOCUMENTATION ---"""

    def _was_recently_flagged(self, channel_id: int, user_id: int) -> bool:
        key = (channel_id, user_id)
        if key not in self.recently_flagged:
            return False
        return datetime.utcnow() - self.recently_flagged[key] < timedelta(minutes=REFLAG_COOLDOWN_MINUTES)

    def _mark_flagged(self, channel_id: int, user_id: int):
        self.recently_flagged[(channel_id, user_id)] = datetime.utcnow()

    def _is_disengaging(self, content: str) -> bool:
        low = content.strip().lower()
        if low in DISENGAGE_COMMANDS:
            return True
        if len(content) < 80:
            return any(p.search(content) for p in DISENGAGE_PATTERNS)
        return False

    async def _disengage(self, message: discord.Message):
        self.conversations.clear(message.channel.id, message.author.id)
        import random
        replies = [
            "Glad I could help! Feel free to ping me anytime 👋",
            "No problem! Come back if you have more questions 😊",
            "Happy to help! Good luck with Bankr 🚀",
            "Anytime! Feel free to tag me if anything else comes up.",
        ]
        await message.reply(random.choice(replies), mention_author=False)
        log.info(f"Disengaged from {message.author} in #{message.channel.name}")

    def _clean_content(self, message: discord.Message) -> str:
        content = message.content
        for mention in message.mentions:
            content = content.replace(f"<@{mention.id}>", "").replace(f"<@!{mention.id}>", "")
        return content.strip()

    # ── Message Routing ────────────────────────────────────────────────────

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return
        if message.id in self._handled_message_ids:
            return

        is_mentioned = self.user in message.mentions
        is_reply_to_bot = (
            message.reference
            and message.reference.resolved
            and isinstance(message.reference.resolved, discord.Message)
            and message.reference.resolved.author == self.user
        )
        has_active_convo = self.conversations.has_active_conversation(
            message.channel.id, message.author.id
        )

        # ── Case 1: Direct mention or reply to bot ──────────────────────
        if is_mentioned or is_reply_to_bot:
            if has_active_convo and self._is_disengaging(message.content):
                await self._disengage(message)
                return

            clean = self._clean_content(message)

            if not has_active_convo and clean and not detect_support_intent(clean)[0]:
                self._handled_message_ids.add(message.id)
                has_korean = bool(re.search(r'[\uac00-\ud7af]', clean))
                has_chinese = bool(re.search(r'[\u4e00-\u9fff]', clean))
                if has_korean:
                    redirect_msg = (
                        "안녕하세요! 저는 Bankr 지원 봇입니다 😊 "
                        "스왑, 지갑, 토큰 발행, API 등 Bankr 플랫폼에 관한 질문을 도와드립니다. "
                        "무엇을 도와드릴까요?"
                    )
                elif has_chinese:
                    redirect_msg = (
                        "你好！我是 Bankr 支持机器人 😊 "
                        "我专门解答关于 Bankr 平台的问题，包括代币兑换、钱包、代币发行、API 等。"
                        "有什么我可以帮你的吗？"
                    )
                else:
                    redirect_msg = (
                        "Hey! I'm the Bankr support bot — I'm here to help with questions about the platform. "
                        "Feel free to ask me anything about swaps, wallets, token launches, the API, or anything else Bankr-related! 😊"
                    )
                await message.reply(redirect_msg, mention_author=False)
                log.info(f"Non-support mention from {message.author}, sent polite redirect")
                return

            self._handled_message_ids.add(message.id)
            await self._handle_support_message(message)
            return

        # ── Case 2: Active conversation ─────────────────────────────────
        if has_active_convo:
            if self._is_disengaging(message.content):
                await self._disengage(message)
                return

            flagged, score = detect_support_intent(message.content)
            if score < 1:
                log.info(
                    f"Active convo with {message.author} but off-topic (score={score}), "
                    f"staying silent: {message.content[:60]}"
                )
                return

            self._handled_message_ids.add(message.id)
            await self._handle_support_message(message)
            return

        # ── Case 3: Passive monitoring ───────────────────────────────────
        if MONITORED_CHANNEL_IDS and message.channel.id not in MONITORED_CHANNEL_IDS:
            return

        if self._was_recently_flagged(message.channel.id, message.author.id):
            return

        flagged, score = detect_support_intent(message.content)
        if flagged:
            log.info(f"Support intent (score={score}) from {message.author}: {message.content[:80]}")
            self._mark_flagged(message.channel.id, message.author.id)
            self._handled_message_ids.add(message.id)
            await self._send_proactive_offer(message)

    async def _send_proactive_offer(self, message: discord.Message):
        try:
            await asyncio.sleep(1)
            greeting = f"Hey {message.author.mention}! 👋 I'm the Bankr support bot — let me help with that:"
            await message.reply(greeting, mention_author=False)
            await self._handle_support_message(message)
        except discord.errors.DiscordServerError as e:
            log.warning(f"Discord server error in proactive offer (skipping): {e}")
        except Exception as e:
            log.error(f"Unexpected error in proactive offer: {e}")

    async def _handle_support_message(self, message: discord.Message):
        content = self._clean_content(message)

        if not content:
            await message.reply("Hey! What can I help you with? 😊", mention_author=False)
            return

        self.conversations.add_message(message.channel.id, message.author.id, "user", content)

        try:
            async with message.channel.typing():
                relevant_docs = await self.docs.query(content)
                system        = self._build_system_prompt(relevant_docs)
                history       = self.conversations.get_history(message.channel.id, message.author.id)
                response      = await self.ollama.chat(history, system=system)
        except discord.errors.DiscordServerError as e:
            log.warning(f"Discord 503 on typing indicator, continuing without it: {e}")
            relevant_docs = await self.docs.query(content)
            system        = self._build_system_prompt(relevant_docs)
            history       = self.conversations.get_history(message.channel.id, message.author.id)
            response      = await self.ollama.chat(history, system=system)

        self.conversations.add_message(message.channel.id, message.author.id, "assistant", response)

        if len(response) <= 1900:
            await message.reply(response, mention_author=False)
        else:
            chunks = [response[i:i + 1900] for i in range(0, len(response), 1900)]
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await message.reply(chunk, mention_author=False)
                else:
                    await message.channel.send(chunk)

        log.info(f"Responded to {message.author} in #{message.channel.name}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    if not DISCORD_TOKEN:
        raise ValueError("DISCORD_TOKEN not set in .env file")
    if not OLLAMA_API_KEY:
        log.warning("OLLAMA_API_KEY not set — required for Ollama cloud models!")

    bot = BankrSupportBot()
    bot.run(DISCORD_TOKEN)


if __name__ == "__main__":
    main()
