import logging
import math
from typing import Tuple

import httpx
import numpy as np
import requests
import torch
from attr import dataclass
from fastrtc import PauseDetectionModel
from fastrtc.speech_to_text.stt_ import STTModel
from fastrtc.utils import AudioChunk, audio_to_float32
from funasr import AutoModel
from numpy.typing import NDArray

from asr.api.api_handler import AvailableModelsResponse, RTCASRRequest
from asr.api.utils import resample_audio
from asr.models.model_interface import ModelDetail

asr_logger = logging.getLogger("ASR")


class RTCASRAdapter(STTModel):

    def __init__(self, base_url: str, model_name: str):
        self.model_name = model_name
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url)
        self.model_info = self.get_model_info()

    def get_model_info(self) -> ModelDetail:
        response = requests.get(self.base_url + "/v1/asr/models/info")

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

        self.expected_lang = "auto"
        for lang in self.model_info.languages:
            # Chinese by default
            if lang == "zh" or lang == "Chinese":
                self.expected_lang = lang
                break

        raise AssertionError(
            "Can not get the model info, the model might not be initialized."
        )

    def stt(self, audio: Tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        sample_rate, audio_array = audio
        asr_logger.info(
            f"Received audio data: sample_rate={sample_rate}, data_type={type(audio_array)}, shape={getattr(audio_array, 'shape', 'N/A')}"
        )

        try:
            audio_array = audio_to_float32(audio_array)

            if audio_array.ndim > 1:
                audio_array = (
                    audio_array[0]
                    if audio_array.shape[0] < audio_array.shape[1]
                    else audio_array[:, 0]
                )

            payload = RTCASRRequest(
                audio=audio_array,
                sample_rate=sample_rate,
                language=self.expected_lang,
                model_name="SenseVoiceSmall",
            )
            asr_logger.info(f"Processed audio length: {len(audio_array)}")
            response = self.client.post("/v1/asr/transcribe/rtc", json=payload)

            if response.status_code != 200:
                print(f"ASR音频转换文本失败: {response.status_code}")
                raise requests.HTTPError(
                    "ASR failed, check if the ASR service is started."
                )
            return response.text

        except Exception as e:
            asr_logger.error(f"Speech transcription error: {e}")
            return f"Transcription failed: {str(e)}"


@dataclass
class FSMNVadOptions:
    model = "fsmn-vad"
    model_revision = "v2.0.4"


class FSMNVad(PauseDetectionModel):
    def __init__(self):

        if torch.cuda.is_available():
            device = "cuda"
            asr_logger.info("CUDA device detected, using GPU acceleration")
        else:
            device = "cpu"
            asr_logger.info("No CUDA device detected, using CPU")
        self.model = AutoModel(model="fsmn-vad", model_revision="v2.0.4", device=device)

    def warmup(self):
        for _ in range(10):
            dummy_audio = np.zeros(102400, dtype=np.float32)
            self.vad((24000, dummy_audio), None)

    def vad(
        self,
        audio: tuple[int, NDArray[np.float32] | NDArray[np.int16]],
        options: None | FSMNVadOptions,
    ) -> tuple[float, list[AudioChunk]]:
        sample_rate, audio_array = audio
        try:
            audio_array = audio_to_float32(audio_array)
            sample_rate, audio_array = resample_audio(audio_array, sample_rate, 16000)

            if audio_array.ndim > 1:
                audio_array = (
                    audio_array[0]
                    if audio_array.shape[0] < audio_array.shape[1]
                    else audio_array[:, 0]
                )

            result = self.model.generate(input=audio_array, disable_pbar=True)
            return self.convert_to_output(result[0]["value"], sample_rate)
        except Exception:
            return math.inf, []

    def convert_to_output(
        self, inputs: list[list[int]], sample_rate: int
    ) -> tuple[float, list[AudioChunk]]:
        chunks = []
        sum_sample_num = 0
        for input in inputs:
            start_sample_cnt = int(input[0] * sample_rate / 1000)
            end_sample_cnt = int(input[1] * sample_rate / 1000)
            sum_sample_num += end_sample_cnt - start_sample_cnt
            chunks.append(AudioChunk(start=start_sample_cnt, end=end_sample_cnt))
        speaking_duration = sum_sample_num / sample_rate
        return speaking_duration, chunks
