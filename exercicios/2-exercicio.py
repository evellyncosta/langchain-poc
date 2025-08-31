import os
import hashlib
import sqlite3
import threading
import sys
import subprocess
from datetime import datetime
from tkinter import Tk, Label, Text, Button, StringVar, END, DISABLED, NORMAL, filedialog

from dotenv import load_dotenv

# LangChain / Gemini
try:
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
except ImportError:
    # fallback (algunas versões)
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI

# ElevenLabs
from elevenlabs.client import ElevenLabs

# --------- ENV ---------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError("Faltando GOOGLE_API_KEY no .env")
if not ELEVEN_KEY:
    raise RuntimeError("Faltando ELEVENLABS_API_KEY no .env")

# --------- CONFIG ---------
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
MODEL_ID = "eleven_multilingual_v2"
OUTPUT_FORMAT = "mp3_44100_128"
TARGET_LANG = "en"

DB_PATH = "tts_cache.sqlite"
AUDIO_DIR = "audio_cache"
os.makedirs(AUDIO_DIR, exist_ok=True)

# --------- DB helpers ---------
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

def write_mp3(path: str, audio_bytes: bytes):
    with open(path, "wb") as f:
        f.write(audio_bytes)

# --------- LLM & TTS ---------
template = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English:\n```{initial_text}```"
)
llm_en = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
translate_chain = template | llm_en | StrOutputParser()

client = ElevenLabs(api_key=ELEVEN_KEY)

def eleven_convert_to_bytes(text: str) -> bytes:
    stream = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format=OUTPUT_FORMAT,
    )
    chunks = []
    for chunk in stream:
        chunks.append(chunk)
    return b"".join(chunks)

# --------- Plataforma: abrir arquivo ---------
def open_file(path: str):
    if sys.platform.startswith("win"):
        os.startfile(path)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)

# --------- GUI ---------
class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("Tradutor + TTS (cache SQLite)")

        self.status = StringVar()
        self.status.set("Pronto.")

        Label(root, text="Digite o texto para traduzir ao inglês:").pack(anchor="w", padx=10, pady=(10, 4))

        self.input_box = Text(root, height=6, width=70)
        self.input_box.pack(padx=10, pady=4)
        self.input_box.insert(END, "Langchain é um framework para desenvolvimento de aplicações de IA.")

        self.btn_frame = None

        self.translate_button = Button(root, text="Traduzir e gerar áudio", command=self.on_translate_click)
        self.translate_button.pack(padx=10, pady=(6, 2))

        self.save_as_button = Button(root, text="Salvar como…", command=self.on_save_as_click, state=DISABLED)
        self.save_as_button.pack(padx=10, pady=2)

        self.open_button = Button(root, text="Abrir arquivo gerado", command=self.on_open_click, state=DISABLED)
        self.open_button.pack(padx=10, pady=(2, 8))

        self.status_label = Label(root, textvariable=self.status, fg="blue")
        self.status_label.pack(anchor="w", padx=10, pady=(0, 10))

        self.last_mp3_path = None
        self.last_translation = None

    def on_translate_click(self):
        text = self.input_box.get("1.0", END).strip()
        if not text:
            self.status.set("Digite algum texto.")
            return
        # roda em thread pra não travar a UI
        t = threading.Thread(target=self._process_text, args=(text,), daemon=True)
        t.start()

    def on_save_as_click(self):
        if not self.last_mp3_path:
            return
        target = filedialog.asksaveasfilename(
            defaultextension=".mp3",
            filetypes=[("MP3 files", "*.mp3")],
            initialfile=os.path.basename(self.last_mp3_path)
        )
        if target:
            try:
                with open(self.last_mp3_path, "rb") as src, open(target, "wb") as dst:
                    dst.write(src.read())
                self.status.set(f"Arquivo copiado para: {target}")
            except Exception as e:
                self.status.set(f"Erro ao salvar: {e}")

    def on_open_click(self):
        if self.last_mp3_path and os.path.exists(self.last_mp3_path):
            open_file(self.last_mp3_path)

    def _process_text(self, source_text: str):
        try:
            self.set_status("Verificando cache…")
            conn = init_db()
            key = make_key(source_text)
            mp3_path = os.path.join(AUDIO_DIR, f"{key}.mp3")

            cached = get_cached(conn, key)
            if cached:
                self.last_translation = cached["translated_text"]
                if not os.path.exists(mp3_path):
                    # reconstruir arquivo a partir do blob
                    self.set_status("Restaurando áudio do cache…")
                    write_mp3(mp3_path, cached["audio_blob"])
                self.last_mp3_path = mp3_path
                self.set_status(f"[CACHE] Tradução pronta. Arquivo: {mp3_path}")
                self.enable_post_buttons()
                return

            # Miss: precisa gerar
            self.set_status("Traduzindo…")
            translated = translate_chain.invoke({"initial_text": source_text})
            self.last_translation = translated

            self.set_status("Gerando áudio…")
            audio_bytes = eleven_convert_to_bytes(translated)

            save_cache(conn, key, source_text, translated, audio_bytes)
            write_mp3(mp3_path, audio_bytes)
            self.last_mp3_path = mp3_path

            self.set_status(f"Concluído. Arquivo salvo em: {mp3_path}")
            self.enable_post_buttons()

        except Exception as e:
            self.set_status(f"Erro: {e}")

    def set_status(self, msg: str):
        # Atualiza a UI de forma thread-safe
        self.root.after(0, lambda: self.status.set(msg))

    def enable_post_buttons(self):
        def _enable():
            self.save_as_button.config(state=NORMAL)
            self.open_button.config(state=NORMAL)
        self.root.after(0, _enable)

def main():
    root = Tk()
    app = App(root)
    root.mainloop()

if __name__ == "__main__":
    main()
