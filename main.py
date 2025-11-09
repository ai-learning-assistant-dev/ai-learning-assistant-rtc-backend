import json
from typing import Generator, Union

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastrtc import AdditionalOutputs, ReplyOnPause, Stream
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from funasr_stt.stt_adapter import LocalFunASR
from kokoro_tts.tts_adapter import get_kokoro_v11_zh_model


class Settings(BaseSettings):
    llm_stream_url: str = "http://localhost:3000/api/ai-chat/chat/stream"
    app_port: int = 8989
    app_host: str = "0.0.0.0"

    class Config:
        env_file = ".env"


settings = Settings()


class RTCMetaData:
    userId: str = ""
    sectionId: str = ""
    personaId: str = ""
    sessionId: str = ""


rtc_metadata = RTCMetaData()


def llm_response(message: str) -> Generator[str, None, None]:
    resp = requests.post(
        settings.llm_stream_url,
        json={
            "userId": rtc_metadata.userId,
            "sectionId": rtc_metadata.sectionId,
            "personaId": rtc_metadata.personaId,
            "sessionId": rtc_metadata.sessionId,
            "message": message,
        },
        stream=True,
    )

    if resp.ok is False:
        raise HTTPException(status_code=resp.status_code)

    for chunk in resp.iter_content(chunk_size=None):
        if chunk:
            try:
                decode_chunk = chunk.decode("utf-8")
                yield decode_chunk
            except UnicodeDecodeError as e:
                print(f"Decode error: {e}, chunk: {chunk[:100]}")
                yield ""


# ASR - FunASR or whisper.cpp
stt_model = LocalFunASR()
tts_model = get_kokoro_v11_zh_model()


def realtime_conversation(audio):
    message = stt_model.stt(audio).strip()
    # 排除掉一个字的情况（一个字一般有标点符号所以长度为2）
    if not message or len(message) < 3:
        return

    # print("REQUEST:", message)
    yield AdditionalOutputs(message)

    response = llm_response(message)

    result = ""
    buffer = ""
    timestamp = 0

    # print("RESPONSE: ", end="")
    for delta in response:
        buffer += delta
        result += delta

        # 碰到句号或停顿
        should_flush_by_punctuation = buffer.endswith(
            (
                "。",
                "，",
                ".",
                ",",
                "!",
                "?",
                " ",
                "！",
                "？",
                "…",
                "—",
                ")",
                "）",
                "”",
                "\n",
            )
        )

        # 至少两个字才会开始读，不然就很容易读出来效果很奇怪
        if len(buffer.strip()) >= 3 and should_flush_by_punctuation:
            # print("[TIME]:", timestamp, buffer, flush=True)
            yield AdditionalOutputs(timestamp, buffer)
            for chunk in tts_model.stream_tts_sync(buffer):
                timestamp += len(chunk[1]) / chunk[0]
                yield chunk
            buffer = ""  # 清空继续积累

    if buffer:
        # print("[TIME]:", timestamp, buffer, flush=True)
        yield AdditionalOutputs(timestamp, buffer)
        for chunk in tts_model.stream_tts_sync(buffer):
            yield chunk


stream = Stream(
    ReplyOnPause(realtime_conversation),
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

    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
