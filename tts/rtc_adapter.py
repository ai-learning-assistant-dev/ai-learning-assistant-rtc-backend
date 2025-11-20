import asyncio
import logging
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from fastrtc.text_to_speech.tts import TTSModel, TTSOptions

from tts.models.model_interface import RTCTTSModelInterface

logging.basicConfig(
    level=logging.INFO,
    format="[TTS]: %(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FastRTCTTSOptions(TTSOptions):
    voice: str = "zf_001"
    speed: float = 1.0
    lang: str = "zh"
    join_sentences: bool = True


class FastRTCTTSModel(TTSModel):

    def __init__(
        self,
        model: RTCTTSModelInterface,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model
        self.warmup_model()

    def warmup_model(self):
        try:
            logger.info("Warming up model with test text...")
            _ = self.model.stream_synthesize("测试", "zf_001", 1.0)
            logger.info("Model warmup completed")
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}, continuing anyway")

    def _speed_callable(self, len_ps: int) -> float:
        speed = 0.8
        if len_ps <= 83:
            speed = 1
        elif len_ps < 183:
            speed = 1 - (len_ps - 83) / 500
        return speed * 1.1

    def tts(
        self, text: str, options: FastRTCTTSOptions | None = None
    ) -> tuple[int, NDArray[np.float32]]:
        options = options or FastRTCTTSOptions()

        sample_rate = self.model.get_model_info().sample_rate
        if not text.strip():
            return sample_rate, np.array([], dtype=np.float32)

        wav = self.model.synthesize(text, options.voice, self._speed_callable)
        if wav is None:
            return sample_rate, np.array([], dtype=np.float32)

        if torch.is_tensor(wav):
            wav = wav.cpu().numpy()

        wav = wav.astype(np.float32)

        return sample_rate, wav

    async def stream_tts(
        self, text: str, options: FastRTCTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.float32]], None]:
        options = options or FastRTCTTSOptions()

        sample_rate = self.model.get_model_info().sample_rate
        if not text.strip():
            yield (sample_rate, np.array([], dtype=np.float32))
            return

        generator = self.model.stream_synthesize(
            text, options.voice, self._speed_callable
        )

        async for wav in generator:
            if wav is None:
                continue

            if torch.is_tensor(wav):
                wav = wav.cpu().numpy()

            yield (sample_rate, wav.astype(np.float32))

    def stream_tts_sync(
        self, text: str, options: FastRTCTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.float32]], None, None]:
        options = options or FastRTCTTSOptions()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            iterator = self.stream_tts(text, options).__aiter__()
            while True:
                try:
                    yield loop.run_until_complete(iterator.__anext__())
                except StopAsyncIteration:
                    break
        finally:
            loop.close()
