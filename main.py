import json
import logging
import re
import time
from typing import Generator, Union

import click
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastrtc import AdditionalOutputs, AlgoOptions, PauseDetectionModel, ReplyOnPause, Stream
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from requests.models import ReadTimeoutError
from huggingface_hub import hf_hub_download
from fastrtc.pause_detection.silero import SileroVADModel
from functools import lru_cache

logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s]: %(levelname)s | %(asctime)s | %(message)s",
    datefmt="%H:%M:%S",
)

# from funasr_vad.vad_adapter import FSMNVad  # noqa: E402
from funasr_stt.stt_adapter import LocalFunASR  # noqa: E402
from kokoro_tts.tts_adapter import get_kokoro_v11_zh_model  # noqa: E402


class EnvVar(BaseSettings):
    llm_stream_url: str = "http://localhost:3000/api/ai-chat/chat/stream"
    app_port: int = 8989
    app_host: str = "0.0.0.0"
    in_container: str = 'false'

    class Config:
        env_file = ".env"


envs = EnvVar()


class RTCMetaData:
    userId: str = ""
    sectionId: str = ""
    personaId: str = ""
    sessionId: str = ""
    daily: bool = False


rtc_metadata = RTCMetaData()


class LLMStreamError(Exception):
    """Raised when LLM streaming request fails or returns an error status."""


def llm_response(message: str) -> Generator[str, None, None]:
    try:
        session = requests.Session()
        session.trust_env = False
        resp = session.post(
            envs.llm_stream_url,
            json={
                "userId": rtc_metadata.userId,
                "sectionId": rtc_metadata.sectionId,
                "message": message,
                "personaId": rtc_metadata.personaId,
                "sessionId": rtc_metadata.sessionId,
                "useAudio": True,
                "ttsOption": "kokoro",
                "reasoning": False,
                "daily": rtc_metadata.daily,
            },
            stream=True,
            timeout=10,
        )
    except requests.RequestException as e:
        logging.exception("Failed to connect to LLM stream")
        raise LLMStreamError(f"大模型请求失败: {e}")

    if not resp.ok:
        raise LLMStreamError(f"大模型API返回错误: {resp.status_code}")

    last_activity = time.time()

    try:
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                last_activity = time.time()
                try:
                    yield chunk.decode("utf-8")
                except UnicodeDecodeError:
                    logging.exception("Decode error from LLM stream")
                    # yield nothing for this chunk but continue streaming
                    yield ""
            else:
                # If no chunk received for >20s → break
                if time.time() - last_activity > 20:
                    raise LLMStreamError("大模型响应超时")
    except TimeoutError or ConnectionError or ReadTimeoutError as e:
        logging.exception("Connection error from LLM stream")
        raise LLMStreamError(f"大模型响应失败：{e}")
    finally:
        resp.close()


# ASR - FunASR or whisper.cpp
stt_model = LocalFunASR()
tts_model = get_kokoro_v11_zh_model()


def realtime_conversation(audio):
    message = stt_model.stt(audio).strip()
    if not message:
        return

    meaningless_patterns = [
        r"^嗯+。$",
        r"^啊+。$",
        r"^呃+。$",
        r"^哦+。$",
        r"^哎+。$",
        r"^哼+。$",
        r"^哈+。$",
        r"^呵+。$",
        r"^咳+。$",
        r"^我。$",
        r"^。$",
    ]

    for pattern in meaningless_patterns:
        if re.match(pattern, message):
            return

    # print("REQUEST:", message)
    yield AdditionalOutputs(message)

    response = llm_response(message)

    result = ""
    buffer = ""
    timestamp = 0
    break_chars = {
        "。",
        "！",
        "？",
        "!",
        "?",
        "\n",
        "，",
        ",",
        # " ",
        "…",
        "—",
        ")",
        "）",
        "”",
    }

    # print("RESPONSE: ", end="")
    try:
        for delta in response:
            buffer += delta
            result += delta

            # 实时扫描缓冲区中的断句标点
            while True:
                found_break = False
                for i, char in enumerate(buffer):
                    if char in break_chars:
                        # 找到断句点，检查分段长度
                        segment = buffer[: i + 1]
                        if len(segment.strip()) >= 2:  # 最小长度限制
                            yield AdditionalOutputs(timestamp, segment)
                            for chunk in tts_model.stream_tts_sync(segment):
                                timestamp += len(chunk[1]) / chunk[0]
                                yield chunk
                            buffer = buffer[i + 1 :]  # 更新缓冲区
                            found_break = True
                            break  # 重新扫描新的缓冲区

                # 如果没有找到合适的断句点，或者缓冲区太短，退出循环
                if not found_break or len(buffer.strip()) < 2:
                    break
        # 处理剩余内容
        if buffer.strip():
            yield AdditionalOutputs(timestamp, buffer)
            for chunk in tts_model.stream_tts_sync(buffer):
                yield chunk

    except LLMStreamError as e:
        logging.exception("LLM stream error while generating response")
        # 有礼貌地告诉用户出错，并尝试通过 TTS 返回一条短消息
        error_text = f"抱歉，智能助理暂时无法响应，请稍后再试。错误信息：{e}。"
        yield AdditionalOutputs(timestamp, error_text)
        for chunk in tts_model.stream_tts_sync(error_text):
            yield chunk
    except Exception as e:
        logging.exception("Unexpected error during LLM streaming")
        error_text = f"发生未知错误，请稍后重试。错误信息：{e}。"
        yield AdditionalOutputs(timestamp, error_text)
        for chunk in tts_model.stream_tts_sync(error_text):
            yield chunk
    finally:
        response.close()

@lru_cache
def get_silero_model() -> PauseDetectionModel:
    # 解决启动要联网问题
    @staticmethod
    def custom_download_model() -> str:
        if(envs.in_container != 'false'):
            return hf_hub_download(
                repo_id="freddyaboulton/silero-vad", filename="silero_vad.onnx", local_files_only=True
            )
        else:
            return hf_hub_download(
                repo_id="freddyaboulton/silero-vad", filename="silero_vad.onnx"
            )
    SileroVADModel.download_model = custom_download_model
    """Returns the VAD model instance and warms it up with dummy data."""
    # Warm up the model with dummy data

    try:
        import importlib.util

        mod = importlib.util.find_spec("onnxruntime")
        if mod is None:
            raise RuntimeError("Install fastrtc[vad] to use ReplyOnPause")
    except (ValueError, ModuleNotFoundError):
        raise RuntimeError("Install fastrtc[vad] to use ReplyOnPause")
    model = SileroVADModel()
    print(click.style("INFO", fg="green") + ":\t  Warming up VAD model.")
    model.warmup()
    print(click.style("INFO", fg="green") + ":\t  VAD model warmed up.")
    return model


stream = Stream(
    ReplyOnPause(
        realtime_conversation,
        algo_options=AlgoOptions(started_talking_threshold=0.5),
        model=get_silero_model(),
        input_sample_rate=16000,
    ),
    modality="audio",
    mode="send-receive",
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LLMMetaData(BaseModel):
    userId: str
    sectionId: str
    personaId: Union[str, None] = None
    sessionId: str
    daily: bool


@app.post("/webrtc/metadata")
def parse_input(metadata: LLMMetaData):
    rtc_metadata.personaId = (
        metadata.personaId if metadata.personaId is not None else ""
    )
    rtc_metadata.sectionId = metadata.sectionId
    rtc_metadata.sessionId = metadata.sessionId
    rtc_metadata.userId = metadata.userId
    rtc_metadata.daily = metadata.daily or False

    print("获取metadata成功")
    return ""


@app.get("/webrtc/text-stream")
def rtc_text_stream(webrtc_id: str):
    async def output_stream():
        async for output in stream.output_stream(webrtc_id):
            if len(output.args) == 1:
                # 用户输入没有时间戳
                payload = {"type": "request", "timestamp": "", "text": output.args[0]}
            else:
                # 时间戳单位是秒
                payload = {
                    "type": "response",
                    "timestamp": output.args[0],
                    "text": output.args[1],
                }
            print("payload =", payload)
            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")


stream.mount(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=envs.app_host, port=envs.app_port)
