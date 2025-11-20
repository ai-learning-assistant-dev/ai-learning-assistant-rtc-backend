import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
from fastrtc import PauseDetectionModel
from fastrtc.utils import AudioChunk, audio_to_float32
from funasr import AutoModel
from numpy.typing import NDArray

from ..funasr_utils.resample import resample_audio

logging.basicConfig(
    level=logging.INFO,
    format="[VAD]: %(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FSMNVadOptions:
    model = "fsmn-vad"
    model_revision = "v2.0.4"


class FSMNVad(PauseDetectionModel):
    def __init__(self):

        if torch.cuda.is_available():
            device = "cuda:0"
            logger.info("CUDA device detected, using GPU acceleration")
        else:
            device = "cpu"
            logger.info("No CUDA device detected, using CPU")
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
            sample_rate, audio_array = resample_audio(audio_array, sample_rate)

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
