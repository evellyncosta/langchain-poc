import os
import hashlib
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from elevenlabs.client import ElevenLabs

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
assert GOOGLE_API_KEY, "Faltando GOOGLE_API_KEY no .env"
assert ELEVEN_KEY, "Faltando ELEVENLABS_API_KEY no .env"

# --------- Configs fixas (entram no hash) ----------
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
MODEL_ID = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"
TARGET_LANG = "en"  # só p/ compor a chave; estamos “traduzindo para inglês”
# ---------------------------------------------------

DB_PATH = "tts_cache.sqlite"
AUDIO_DIR = "audio_cache"
os.makedirs(AUDIO_DIR, exist_ok=True)

# ---- Banco ----
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            source_text TEXT NOT NULL,
            translated_text TEXT NOT NULL,
            voice_id TEXT NOT NULL,
            model_id TEXT NOT NULL,
            output_format TEXT NOT NULL,
            target_lang TEXT NOT NULL,
            audio_blob BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

def make_key(source_text: str) -> str:
    payload = "|".join([
        source_text.strip(),
        TARGET_LANG,
        VOICE_ID,
        MODEL_ID,
        OUTPUT_FORMAT,
    ]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def get_cached(conn, key: str):
    cur = conn.execute(
        "SELECT translated_text, audio_blob FROM cache WHERE key = ?",
        (key,)
    )
    row = cur.fetchone()
    if row:
        return {"translated_text": row[0], "audio_blob": row[1]}
    return None

def save_cache(conn, key: str, source_text: str, translated_text: str, audio_bytes: bytes):
    conn.execute(
        """INSERT OR REPLACE INTO cache
        (key, source_text, translated_text, voice_id, model_id, output_format, target_lang, audio_blob, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (key, source_text, translated_text, VOICE_ID, MODEL_ID, OUTPUT_FORMAT, TARGET_LANG, sqlite3.Binary(audio_bytes), datetime.utcnow().isoformat())
    )
    conn.commit()

def write_mp3_temp(key: str, audio_bytes: bytes) -> str:
    path = os.path.join(AUDIO_DIR, f"{key}.mp3")
    with open(path, "wb") as f:
        f.write(audio_bytes)
    return path

# ---- LangChain (tradução) ----
template = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English and show me only the translated text, common and concise in only 1 phrase :\n```{initial_text}```"
)
llm_en = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
translate_chain = template | llm_en | StrOutputParser()

# ---- ElevenLabs (TTS) ----
client = ElevenLabs(api_key=ELEVEN_KEY)

def eleven_convert_to_bytes(text: str) -> bytes:
    stream = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format=OUTPUT_FORMAT,
    )
    # stream é um iterador de chunks; juntar em bytes
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
    return b"".join(chunks)

# ---- Função principal ----
def translate_and_tts(source_text: str):
    conn = init_db()
    key = make_key(source_text)

    cached = get_cached(conn, key)
    if cached:
        print("[CACHE] Usando tradução e áudio do SQLite.")
        translated = cached["translated_text"]
        audio_bytes = cached["audio_blob"]
    else:
        print("[MISS] Gerando tradução e áudio…")
        translated = translate_chain.invoke({"initial_text": source_text})
        audio_bytes = eleven_convert_to_bytes(translated)
        save_cache(conn, key, source_text, translated, audio_bytes)

    # Salva em arquivo para tocar (ou use sua lib favorita de playback)
    mp3_path = write_mp3_temp(key, audio_bytes)
    print("Tradução:", translated)
    print("Áudio salvo em:", mp3_path)
    print("Dica: abra o arquivo para ouvir, ou use uma lib como simpleaudio/pydub para tocar diretamente.")

# --------- Exemplo de uso ---------
if __name__ == "__main__":
    texto = "O caboclo samambaia é um caboclo bonito"
    translate_and_tts(texto)
