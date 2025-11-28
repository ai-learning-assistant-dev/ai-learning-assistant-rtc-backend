import json
import re
from typing import Generator, Union

import logging
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastrtc import AdditionalOutputs, AlgoOptions, ReplyOnPause, Stream
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from funasr_stt.stt_adapter import LocalFunASR
from funasr_vad.vad_adapter import FSMNVad
from kokoro_tts.tts_adapter import get_kokoro_v11_zh_model


class EnvVar(BaseSettings):
    llm_stream_url: str = "http://localhost:3000/api/ai-chat/chat/stream"
    app_port: int = 8989
    app_host: str = "0.0.0.0"

    class Config:
        env_file = ".env"


envs = EnvVar()


class RTCMetaData:
    userId: str = ""
    sectionId: str = ""
    personaId: str = ""
    sessionId: str = ""


rtc_metadata = RTCMetaData()


class LLMStreamError(Exception):
    """Raised when LLM streaming request fails or returns an error status."""


logging.basicConfig(level=logging.INFO)


def llm_response(message: str) -> Generator[str, None, None]:
    try:
        resp = requests.post(
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
            },
            stream=True,
            timeout=30,
        )
    except requests.RequestException as e:
        logging.exception("Failed to connect to LLM stream")
        raise LLMStreamError(f"LLM request failed: {e}")

    if resp.ok is False:
        # try to include response body in the error message for debugging
        body = ""
        try:
            body = resp.text
        except Exception:
            body = "<unable to read response body>"
        logging.error("LLM server error %s: %s", resp.status_code, body[:200])
        # raise a domain-specific error so callers can handle it gracefully
        try:
            resp.close()
        except Exception:
            pass
        raise LLMStreamError(f"LLM server returned {resp.status_code}: {body}")

    try:
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                try:
                    decode_chunk = chunk.decode("utf-8")
                    yield decode_chunk
                except UnicodeDecodeError as e:
                    logging.exception("Decode error from LLM stream")
                    # yield nothing for this chunk but continue streaming
                    yield ""
    finally:
        try:
            resp.close()
        except Exception:
            pass


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
        " ",
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
    except LLMStreamError as e:
        logging.exception("LLM stream error while generating response")
        # 有礼貌地告诉用户出错，并尝试通过 TTS 返回一条短消息
        error_text = "抱歉，智能助理暂时无法响应，请稍后再试。"
        yield AdditionalOutputs(timestamp, error_text)
        for chunk in tts_model.stream_tts_sync(error_text):
            yield chunk
        return
    except Exception:
        logging.exception("Unexpected error during LLM streaming")
        error_text = "发生未知错误，请稍后重试。"
        yield AdditionalOutputs(timestamp, error_text)
        for chunk in tts_model.stream_tts_sync(error_text):
            yield chunk
        return

    # 处理剩余内容
    if buffer.strip():
        yield AdditionalOutputs(timestamp, buffer)
        for chunk in tts_model.stream_tts_sync(buffer):
            yield chunk


stream = Stream(
    ReplyOnPause(
        realtime_conversation,
        algo_options=AlgoOptions(started_talking_threshold=0.5),
        # model=FSMNVad(),
        input_sample_rate=16000,
    ),
    modality="audio",
    mode="send-receive",
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://localhost:\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LLMMetaData(BaseModel):
    userId: str
    sectionId: str
    personaId: Union[str, None] = None
    sessionId: str


@app.post("/webrtc/metadata")
def parse_input(metadata: LLMMetaData):
    rtc_metadata.personaId = (
        metadata.personaId if metadata.personaId is not None else ""
    )
    rtc_metadata.sectionId = metadata.sectionId
    rtc_metadata.sessionId = metadata.sessionId
    rtc_metadata.userId = metadata.userId

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
