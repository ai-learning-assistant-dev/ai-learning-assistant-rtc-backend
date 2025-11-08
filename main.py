from typing import Generator, Union

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastrtc import AdditionalOutputs, ReplyOnPause, Stream
from pydantic import BaseModel

from funasr_stt.stt_adapter import LocalFunASR
from kokoro_tts.tts_adapter import get_kokoro_v11_zh_model


class RTCMetaData:
    userId: str = ""
    sectionId: str = ""
    personaId: str = ""
    sessionId: str = ""


rtc_metadata = RTCMetaData()


LLM_STREAM_URL = "http://localhost:3000/api/ai-chat/chat/stream"


def llm_response(message: str) -> Generator[str, None, None]:
    resp = requests.post(
        LLM_STREAM_URL,
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
    if not message or len(message) < 2:
        return

    print("PROMPT:", message)

    response = llm_response(message)

    result = ""
    buffer = ""

    print("RESPONSE:", end="")
    for delta in response:
        buffer += delta
        result += delta

        yield AdditionalOutputs(delta)
        print(delta, end="", flush=True)
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
            )
        )

        if buffer and should_flush_by_punctuation:
            for chunk in tts_model.stream_tts_sync(buffer):
                yield chunk
            buffer = ""  # 清空继续积累

    if buffer:
        yield AdditionalOutputs(buffer)
        for chunk in tts_model.stream_tts_sync(buffer):
            yield chunk

    print()


stream = Stream(
    ReplyOnPause(realtime_conversation),
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
            yield f"data: {output.args[0]}\n\n"

    return StreamingResponse(output_stream(), media_type="text/event-stream")


stream.mount(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8989)
