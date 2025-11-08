from fastrtc import ReplyOnPause, Stream

from funasr_stt.stt_adapter import LocalFunASR
from kokoro_tts.tts_adapter import get_kokoro_v11_zh_model


# ASR - FunASR or whisper.cpp
stt_model = LocalFunASR()
tts_model = get_kokoro_v11_zh_model()


def warmup(_):
    for chunk in tts_model.stream_tts_sync("你好"):
        yield chunk


stream = Stream(
    ReplyOnPause(warmup),
    modality="audio",
    mode="send-receive",
)
