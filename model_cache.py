import numpy as np
import torch
from fastrtc import ReplyOnPause, Stream
from funasr import AutoModel
from kokoro import KModel, KPipeline

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

stt_model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    hub="ms",
    device=device,
    disable_update=True,
)

_model = (
    KModel(
        repo_id="hexgrad/Kokoro-82M-v1.1-zh",
    )
    .to(device)
    .eval()
)

_en_pipeline = KPipeline(
    lang_code="a", repo_id="hexgrad/Kokoro-82M-v1.1-zh", model=False
)


def en_callable(text):
    if text == "Kokoro":
        return "kˈOkəɹO"
    elif text == "Sol":
        return "sˈOl"
    return next(_en_pipeline(text)).phonemes


_zh_pipeline = KPipeline(
    lang_code="z",
    repo_id="hexgrad/Kokoro-82M-v1.1-zh",
    model=_model,
    en_callable=en_callable,
)


def warmup(_):
    test_result = next(_zh_pipeline("测试", "zf_001", speed=1.0))
    wav = test_result.audio

    if wav is None:
        return

    if torch.is_tensor(wav):
        wav = wav.cpu().numpy()

    yield 24000, wav.astype(np.float32)


stream = Stream(
    ReplyOnPause(warmup),
    modality="audio",
    mode="send-receive",
)
