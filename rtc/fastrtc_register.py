import re
from typing import Generator

import logging
import requests
from fastrtc import AdditionalOutputs, AlgoOptions, ReplyOnPause, Stream
from fastrtc.speech_to_text.stt_ import STTModel
from fastrtc.pause_detection.protocol import PauseDetectionModel
from env import envs

from tts.models.model_interface import RTCTTSModelInterface
from tts.rtc_adapter import FastRTCTTSModel


class RTCLLMMetaData:
    userId: str = ""
    sectionId: str = ""
    personaId: str = ""
    sessionId: str = ""
    modelName: str = ""


class LLMStreamError(Exception):
    """Raised when LLM streaming request fails or returns an error status."""


class FastRTCRegister:
    def __init__(
        self,
        tts_model: RTCTTSModelInterface | None = None,
        stt_model: STTModel | None = None,
        vad_model: PauseDetectionModel | None = None,
    ):
        if tts_model is not None:
            self.tts_model = FastRTCTTSModel(tts_model)
        else:
            self.tts_model = None
        self.stt_model = stt_model
        self.vad_model = vad_model
        self.stream = Stream(
            ReplyOnPause(
                self.realtime_conversation,
                algo_options=AlgoOptions(started_talking_threshold=0.5),
                model=vad_model,
                input_sample_rate=16000,
            ),
            modality="audio",
            mode="send-receive",
        )
        self.metadata = RTCLLMMetaData()

    def load_tts_model(self, tts_model: RTCTTSModelInterface):
        self.tts_model = FastRTCTTSModel(tts_model)

    def load_stt_model(self, stt_model: STTModel):
        self.stt_model = stt_model

    def load_vad_model(self, vad_model: PauseDetectionModel):
        self.vad_model = vad_model
        # restart stream with new VAD model
        self.stream = Stream(
            ReplyOnPause(
                self.realtime_conversation,
                algo_options=AlgoOptions(started_talking_threshold=0.5),
                model=vad_model,
                input_sample_rate=16000,
            ),
            modality="audio",
            mode="send-receive",
        )

    def llm_response(self, message: str) -> Generator[str, None, None]:
        if (
            self.metadata.userId == ""
            or self.metadata.sectionId == ""
            or self.metadata.sessionId == ""
        ):
            raise LLMStreamError("LLM metadata is incomplete")
        try:
            resp = requests.post(
                envs.llm_stream_url,
                json={
                    "userId": self.metadata.userId,
                    "sectionId": self.metadata.sectionId,
                    "message": message,
                    "personaId": self.metadata.personaId,
                    "sessionId": self.metadata.sessionId,
                    "useAudio": True,
                    "ttsOption": "kokoro",
                    "daily": False if len(self.metadata.sectionId) else True,
                    "modelName": self.metadata.modelName,
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

    def realtime_conversation(self, audio):
        if self.stt_model is None or self.tts_model is None:
            logging.error("STT model or TTS model is not set in FastRTCRegister")
            return
        message = self.stt_model.stt(audio).strip()
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

        response = self.llm_response(message)

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
                                for chunk in self.tts_model.stream_tts_sync(segment):
                                    timestamp += len(chunk[1]) / chunk[0]
                                    yield chunk
                                buffer = buffer[i + 1 :]  # 更新缓冲区
                                found_break = True
                                break  # 重新扫描新的缓冲区

                    # 如果没有找到合适的断句点，或者缓冲区太短，退出循环
                    if not found_break or len(buffer.strip()) < 2:
                        break
        except LLMStreamError as e:
            logging.exception(f"LLM stream error while generating response: {e}")
            # 告诉用户出错，并尝试通过 TTS 返回一条短消息
            error_text = "抱歉，智能助理暂时无法响应，请稍后再试。"
            yield AdditionalOutputs(timestamp, error_text)
            for chunk in self.tts_model.stream_tts_sync(error_text):
                yield chunk
            return
        except Exception:
            logging.exception("Unexpected error during LLM streaming")
            error_text = "发生未知错误，请稍后重试。"
            yield AdditionalOutputs(timestamp, error_text)
            for chunk in self.tts_model.stream_tts_sync(error_text):
                yield chunk
            return

        # 处理剩余内容
        if buffer.strip():
            yield AdditionalOutputs(timestamp, buffer)
            for chunk in self.tts_model.stream_tts_sync(buffer):
                yield chunk


logging.basicConfig(level=logging.INFO)


fastrtc_register = FastRTCRegister()
