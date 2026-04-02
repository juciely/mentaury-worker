import os
import uuid
import asyncio
import tempfile
import ffmpeg
from pathlib import Path
from dotenv import load_dotenv

import whisper
import torch
from pyannote.audio import Pipeline
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import httpx
from supabase import create_client

load_dotenv()

app = FastAPI()

# Config
HF_TOKEN = os.getenv("HF_TOKEN")
WORKER_SECRET = os.getenv("VPS_WORKER_SECRET")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Inicializa clientes
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Carrega modelos uma vez na inicialização (não a cada request)
print("Carregando Whisper Small...")
whisper_model = whisper.load_model("small")
print("Whisper carregado.")

print("Carregando pyannote...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)
print("pyannote carregado.")

# Diretório temporário para áudios
TEMP_DIR = Path("./temp_audio")
TEMP_DIR.mkdir(exist_ok=True)


# --- Modelos de request ---

class JobRequest(BaseModel):
    job_id: str
    file_path: str
    source: str  # 'upload' | 'whatsapp'


# --- Autenticação ---

def verify_secret(authorization: str = Header(...)):
    if authorization != f"Bearer {WORKER_SECRET}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# --- Utilitários ---

def download_audio(file_path: str, dest: Path) -> Path:
    """Baixa áudio do Supabase Storage para disco local."""
    response = supabase.storage.from_("audios").download(file_path)
    dest.write_bytes(response)
    return dest


def extract_audio(input_path: Path, output_path: Path) -> Path:
    """Extrai áudio de vídeo ou converte para WAV."""
    (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), ac=1, ar=16000)  # mono, 16kHz — ideal pro Whisper
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Divide texto em chunks com overlap."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


async def notify_supabase_embeddings(job_id: str, mentor_id: str, transcriptions: list):
    """Chama Edge Function generate-embeddings no Supabase."""
    supabase_url = os.getenv("SUPABASE_URL")
    async with httpx.AsyncClient() as client:
        await client.post(
            f"{supabase_url}/functions/v1/generate-embeddings",
            json={
                "job_id": job_id,
                "mentor_id": mentor_id,
                "transcriptions": transcriptions
            },
            headers={
                "Authorization": f"Bearer {WORKER_SECRET}",
                "Content-Type": "application/json"
            },
            timeout=120.0
        )


# --- Endpoint principal ---

@app.post("/jobs")
async def process_job(job: JobRequest, authorization: str = Header(...)):
    verify_secret(authorization)

    job_id = job.job_id
    tmp_id = str(uuid.uuid4())

    # Atualiza status
    supabase.table("processing_jobs").update(
        {"status": "diarizing"}
    ).eq("id", job_id).execute()

    try:
        # Busca dados do job
        result = supabase.table("processing_jobs").select(
            "mentor_id, mentor_speaker_id, source, speaker_confirmed"
        ).eq("id", job_id).single().execute()
        job_data = result.data

        # Caminhos temporários
        raw_path = TEMP_DIR / f"{tmp_id}_raw"
        wav_path = TEMP_DIR / f"{tmp_id}.wav"

        # Download do arquivo
        download_audio(job.file_path, raw_path)

        # Converte para WAV mono 16kHz
        extract_audio(raw_path, wav_path)

        transcriptions = []

        # --- FLUXO WHATSAPP: falante único, pula diarização ---
        if job.source == "whatsapp":
            supabase.table("processing_jobs").update(
                {"status": "transcribing"}
            ).eq("id", job_id).execute()

            result = whisper_model.transcribe(str(wav_path), language="pt")
            transcriptions.append({
                "speaker": "SPEAKER_0",
                "texto": result["text"],
                "start_time": 0.0,
                "end_time": 0.0
            })

        # --- FLUXO UPLOAD: diarização + transcrição por segmento ---
        else:
            # Diarização
            diarization = diarization_pipeline(str(wav_path))

            # Verifica se mentor já confirmou o speaker
            mentor_speaker = job_data.get("mentor_speaker_id")
            speaker_confirmed = job_data.get("speaker_confirmed", False)

            if not speaker_confirmed:
                # Gera preview para confirmação: 3 trechos de cada speaker
                speakers_preview = {}
                for turn, _, speaker in diarization.itertracks(yield_label=True):
                    if speaker not in speakers_preview:
                        speakers_preview[speaker] = []
                    if len(speakers_preview[speaker]) < 3:
                        # Transcreve só esse trecho pra preview
                        segment_result = whisper_model.transcribe(
                            str(wav_path),
                            language="pt",
                            clip_timestamps=[turn.start, turn.end]
                        )
                        speakers_preview[speaker].append({
                            "start": turn.start,
                            "end": turn.end,
                            "texto": segment_result["text"]
                        })

                # Salva preview e aguarda confirmação
                supabase.table("processing_jobs").update({
                    "status": "pending",
                    "error_message": f"AGUARDANDO_CONFIRMACAO:{str(speakers_preview)}"
                }).eq("id", job_id).execute()

                # Limpa temporários e retorna
                raw_path.unlink(missing_ok=True)
                wav_path.unlink(missing_ok=True)
                return {"status": "awaiting_confirmation", "job_id": job_id}

            # Speaker confirmado — transcreve apenas segmentos do mentor
            supabase.table("processing_jobs").update(
                {"status": "transcribing"}
            ).eq("id", job_id).execute()

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker != mentor_speaker:
                    continue  # ignora falas que não são do mentor

                segment_result = whisper_model.transcribe(
                    str(wav_path),
                    language="pt",
                    clip_timestamps=[turn.start, turn.end]
                )

                if segment_result["text"].strip():
                    transcriptions.append({
                        "speaker": speaker,
                        "texto": segment_result["text"].strip(),
                        "start_time": turn.start,
                        "end_time": turn.end
                    })

        # Limpa temporários
        raw_path.unlink(missing_ok=True)
        wav_path.unlink(missing_ok=True)

        # Atualiza status e envia pra Edge Function de embeddings
        supabase.table("processing_jobs").update(
            {"status": "embedding"}
        ).eq("id", job_id).execute()

        await notify_supabase_embeddings(
            job_id=job_id,
            mentor_id=job_data["mentor_id"],
            transcriptions=transcriptions
        )

        return {"status": "ok", "segments": len(transcriptions)}

    except Exception as e:
        # Registra erro no job
        supabase.table("processing_jobs").update({
            "status": "error",
            "error_message": str(e)
        }).eq("id", job_id).execute()

        # Limpa temporários se existirem
        for p in [raw_path, wav_path]:
            try:
                p.unlink(missing_ok=True)
            except:
                pass

        raise HTTPException(status_code=500, detail=str(e))


# --- Health check ---

@app.get("/health")
def health():
    return {
        "status": "ok",
        "whisper": "loaded",
        "pyannote": "loaded"
    }
