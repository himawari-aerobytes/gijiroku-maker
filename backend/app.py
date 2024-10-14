# app.py
import os
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AzureOpenAI
import json
import logging
from typing import List, Dict, Any
import io
import wave
import time
from pydantic import BaseModel
from google.cloud.speech_v1p1beta1.services.speech import SpeechAsyncClient
from google.cloud.speech_v1p1beta1 import types
from starlette.websockets import WebSocketState
from dotenv import load_dotenv

BUFFER_TIME_SECONDS = 30

app = FastAPI()

# dotenvを読み込む
load_dotenv()

# Google Cloud の認証情報のパスを環境変数から取得
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

audio_buffer = bytearray()

# CORS設定
origins = [
    "http://localhost:3000",
    # 他の許可するオリジンを追加
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure OpenAIの設定を環境変数から取得
whisper_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_WHISPER_ENDPOINT"),
    api_version="2024-06-01",
    api_key=os.getenv("AZURE_API_KEY")
)

gpt_client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_GPT_ENDPOINT"),
    api_version="2024-02-15-preview",
    api_key=os.getenv("AZURE_API_KEY")
)

# Google Speech-to-Text 非同期クライアントの初期化
speech_client = SpeechAsyncClient()

# ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 接続管理クラス
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New connection. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Connection closed. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        try:
            await websocket.send_json(message)
            logger.debug(f"Sent message: {json.dumps(message)}")
        except WebSocketDisconnect:
            logger.warning("WebSocket disconnected while sending message")
            self.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections.copy():
            try:
                await connection.send_json(message)
                logger.debug(f"Broadcasted message: {json.dumps(message)}")
            except WebSocketDisconnect:
                logger.warning("WebSocket disconnected during broadcast")
                self.disconnect(connection)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    logger.info("INFO:app:New connection. Total connections: 1")
    last_send_time = time.time()

    try:
        # Speech-to-Text の設定
        config = types.RecognitionConfig(
            encoding=types.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="ja-JP",
        )
        streaming_config = types.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
        )

        # 非同期ジェネレータ関数
        async def request_generator():
            nonlocal last_send_time
            yield types.StreamingRecognizeRequest(streaming_config=streaming_config)
            try:
                while True:
                    data = await websocket.receive_bytes()
                    logger.debug(f"Received audio data of size: {len(data)} bytes")

                    # 音声データをバッファに追加
                    audio_buffer.extend(data)

                    # Google Speech-to-Textにストリーミングリクエストを送信
                    yield types.StreamingRecognizeRequest(audio_content=data)

                    # Whisper用のバッファデータが一定時間以上たまったら処理
                    current_time = time.time()
                    if current_time - last_send_time >= BUFFER_TIME_SECONDS:
                        asyncio.create_task(handle_whisper_transcription(websocket))
                        last_send_time = current_time
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected during data reception.")
            except Exception as e:
                logger.error(f"Error receiving data: {e}")
                raise e

        # Whisperのバッチ処理関数
        async def handle_whisper_transcription(websocket: WebSocket):
            if not audio_buffer:
                return
            transcription = await transcribe_audio(bytes(audio_buffer))
            await manager.send_personal_message({"type": "transcription", "text": transcription}, websocket)
            audio_buffer.clear()

        # ストリーミング認識の実行（非同期ジェネレータを使用）
        responses = await speech_client.streaming_recognize(requests=request_generator())

        async for response in responses:
            for result in response.results:
                if result.is_final:
                    # pass
                    transcript = result.alternatives[0].transcript
                    logger.info(f"Final transcript: {transcript}")
                    await manager.send_personal_message({"type": "google_transcription", "text": transcript}, websocket)
                else:
                    interim = result.alternatives[0].transcript
                    logger.info(f"Interim transcript: {interim}")
                    await manager.send_personal_message({"type": "immediate", "text": interim}, websocket)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in websocket_endpoint: {e}")
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.send_json({"type": "error", "message": str(e)})
    finally:
        manager.disconnect(websocket)

async def transcribe_audio(audio_data: bytes) -> str:
    try:
        sample_rate = 16000
        channels = 1
        wav_file = pcm_to_wav(audio_data, sample_rate, channels)
        wav_file.name = "audio.wav"

        # Whisper APIを呼び出す
        response = whisper_client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_file,
            language="ja",
        )
        return response.text
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {e}")
        raise HTTPException(status_code=500, detail="音声認識中にエラーが発生しました。")

# TranscriptRequest モデル
class TranscriptRequest(BaseModel):
    transcript: str

# 議事録生成関数
async def generate_minutes(transcript: str):
    prompt = f"以下は会議の文字起こしです。これを基に議事録を作成してください：\n\n{transcript}"
    format = """ 以下のフォーマットに従って議事録を作成してください。わからない部分は、曖昧に回答せず、不明と記述するようにしてください。
                # 議題
                議題を簡潔に入力します。
                # 参加者
                参加者を入力します。不明な場合は不明と明記してください。
                # 依頼事項
                依頼事項を入力します。他部署の関係者に依頼すべきことはここに記入してください。
                # 決定事項
                決定事項を入力します。誰が何をいつまでにするか明記してください。締切や誰がが不明な場合は、不明と明記してください。
                # 議事内容
                議事詳細を入力します。誰が発言したかを（）内に示してください。不明な場合は、不明と明記してください。
            """
    try:
        response = gpt_client.chat.completions.create(
            model=None,
            messages=[
                {"role": "system", "content": "あなたは金融業界のITシステムに関する議事録作成者です。"},
                {"role": "assistant", "content": format},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            top_p=0.8,
            presence_penalty=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating minutes: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="議事録の生成中にエラーが発生しました。")

# 議事録生成エンドポイント
@app.post("/generate_minutes")
async def generate_minutes_endpoint(request: TranscriptRequest):
    logger.info("Generating minutes")
    try:
        minutes = await generate_minutes(request.transcript)
        await manager.broadcast({"type": "minutes", "text": minutes})
        return {"minutes": minutes}
    except HTTPException as e:
        logger.error(f"Error in generate_minutes_endpoint: {e}", exc_info=True)
        return {"error": str(e.detail)}

def pcm_to_wav(pcm_data: bytes, sample_rate: int, num_channels: int) -> io.BytesIO:
    wav_io = io.BytesIO()
    with wave.open(wav_io, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    wav_io.seek(0)
    return wav_io
