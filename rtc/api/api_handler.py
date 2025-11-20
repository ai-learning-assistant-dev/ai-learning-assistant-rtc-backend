# conform with backend LLM API
import json
from typing import Union
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from api import app
from ..fastrtc_register import fastrtc_register


class LLMMetaData(BaseModel):
    userId: str
    sectionId: str
    personaId: Union[str, None] = None
    sessionId: str
    modelName: Union[str, None] = None


@app.post("/webrtc/metadata")
def parse_input(metadata: LLMMetaData):
    fastrtc_register.metadata.personaId = (
        metadata.personaId if metadata.personaId is not None else ""
    )
    fastrtc_register.metadata.modelName = (
        metadata.modelName if metadata.modelName is not None else ""
    )
    fastrtc_register.metadata.sectionId = metadata.sectionId
    fastrtc_register.metadata.sessionId = metadata.sessionId
    fastrtc_register.metadata.userId = metadata.userId

    print("获取metadata成功")
    return ""


@app.get("/webrtc/text-stream")
def rtc_text_stream(webrtc_id: str):
    async def output_stream():
        async for output in fastrtc_register.stream.output_stream(webrtc_id):
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


fastrtc_register.stream.mount(app)
