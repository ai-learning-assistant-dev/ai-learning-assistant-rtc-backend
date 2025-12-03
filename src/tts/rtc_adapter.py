import logging
from collections.abc import AsyncGenerator, Generator
from dataclasses import dataclass

import httpx
import numpy as np
import requests
from fastrtc.text_to_speech.tts import TTSOptions
from numpy.typing import NDArray

from tts.api.api_handler import AvailableModelsResponse, RTCTTSRequest
from tts.models.model_interface import ModelDetail

tts_logger = logging.getLogger("TTS")


@dataclass
class FastRTCTTSOptions(TTSOptions):
    voice: str = "zf_001"
    speed: float = 1.0
    lang: str = "zh"
    join_sentences: bool = True


class RTCTTSAdapter:

    def __init__(
        self,
        base_url: str,
        model_name: str,
    ):
        self.base_url = base_url
        self.model_name = model_name
        self.async_client = httpx.AsyncClient(base_url=base_url)
        self.sync_client = httpx.Client(base_url=base_url)
        self.model_info = self.get_model_info()
        self.warmup_model()

    def get_model_info(self) -> ModelDetail:
        response = requests.get(
            self.base_url + "/v1/tts/models/info/rtc"
        )  # only get rtc tts models

        if response.status_code != 200:
            print(f"获取模型信息失败: {response.status_code}")
            raise requests.HTTPError(
                "Can not get the model info, check if the TTS service is started."
            )

        data = AvailableModelsResponse(**response.json())
        models = data.models

        for model in models:
            if model.model_name != self.model_name:
                continue
            return model
        raise AssertionError(
            "Can not get the model info, the model might not be initialized."
        )

    def warmup_model(self):
        try:
            payload = RTCTTSRequest(
                input="你好，这是一个测试文本",
                voice=self.model_info.voices[0].name,  # 可能不存在
                speed=1.0,
                model=self.model_name,
            )
            with self.sync_client.stream(
                "POST", "/v1/tts/synthesize/rtc", json=payload
            ) as response:
                response.raise_for_status()

        except Exception as e:
            tts_logger.warning(f"Model warmup failed: {e}, continuing anyway")

    @staticmethod
    def _speed_callable(len_ps: int) -> float:
        speed = 0.8
        if len_ps <= 83:
            speed = 1
        elif len_ps < 183:
            speed = 1 - (len_ps - 83) / 500
        return speed * 1.1

    def tts(
        self, text: str, options: FastRTCTTSOptions | None = None
    ) -> tuple[int, NDArray[np.int16]]:
        options = options or FastRTCTTSOptions()

        sample_rate = self.model_info.sample_rate
        if not text.strip():
            return sample_rate, np.array([], dtype=np.int16)

        payload = RTCTTSRequest(
            input=text,
            voice=options.voice,
            speed=0,  # 0 is a special value, assigning it TTS will uses the auto speed callback
            model=self.model_name,
        )
        audio_chunks = []
        try:
            with self.sync_client.stream(
                "POST", "/v1/tts/synthesize/rtc", json=payload
            ) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes():
                    audio_data = np.frombuffer(chunk, dtype=np.int16)
                    audio_chunks.append(audio_data)
        except Exception as e:
            tts_logger.warning(f"RTC TTS failed: {e}, returning empty audio chunk")
            return sample_rate, np.array([], dtype=np.int16)

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
            return sample_rate, full_audio
        return sample_rate, np.array([], dtype=np.int16)

    async def stream_tts(
        self, text: str, options: FastRTCTTSOptions | None = None
    ) -> AsyncGenerator[tuple[int, NDArray[np.int16]], None]:
        options = options or FastRTCTTSOptions()

        sample_rate = self.model_info.sample_rate
        if not text.strip():
            yield (sample_rate, np.array([], dtype=np.int16))
            return

        payload = RTCTTSRequest(
            input=text,
            voice=options.voice,
            speed=1.0,
            model=self.model_name,
        )

        try:
            async with self.async_client.stream(
                "POST", "/v1/tts/synthesize/rtc", json=payload
            ) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield sample_rate, np.frombuffer(chunk, dtype=np.int16)
        except Exception as e:
            tts_logger.error(f"Stream TTS处理出错: {e}")
            yield sample_rate, np.array([], dtype=np.int16)

    def stream_tts_sync(
        self, text: str, options: FastRTCTTSOptions | None = None
    ) -> Generator[tuple[int, NDArray[np.int16]], None, None]:
        options = options or FastRTCTTSOptions()
        sample_rate = self.model_info.sample_rate
        payload = RTCTTSRequest(
            input=text,
            voice=options.voice,
            speed=1.0,
            model=self.model_name,
        )
        with self.sync_client.stream(
            "POST", "/v1/tts/synthesize/rtc", json=payload
        ) as response:
            response.raise_for_status()

            for chunk in response.iter_bytes():
                if chunk:
                    yield sample_rate, np.frombuffer(chunk, dtype=np.int16)
