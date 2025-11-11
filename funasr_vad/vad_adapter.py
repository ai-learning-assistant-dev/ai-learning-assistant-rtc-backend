import math
import tempfile
import wave
from dataclasses import dataclass

import numpy as np
from fastrtc import PauseDetectionModel
from fastrtc.utils import AudioChunk, audio_to_int16
from funasr import AutoModel
from numpy.typing import NDArray


@dataclass
class FSMNVadOptions:
    model = "fsmn-vad"
    model_revision = "v2.0.4"


class FSMNVad(PauseDetectionModel):
    def __init__(self):
        self.model = AutoModel(model="fsmn-vad", model_revision="v2.0.4")

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
            audio_array = audio_to_int16(audio_array)
            if audio_array.ndim > 1:
                audio_array = (
                    audio_array[0]
                    if audio_array.shape[0] < audio_array.shape[1]
                    else audio_array[:, 0]
                )

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name

                with wave.open(tmp_path, "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_array.tobytes())

            result = self.model.generate(input=tmp_path, disable_pbar=True)
            return self.convert_to_output(result[0]['value'], sample_rate)
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
