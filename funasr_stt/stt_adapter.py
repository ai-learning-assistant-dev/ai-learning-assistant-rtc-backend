import logging
from typing import Tuple

import numpy as np
import torch
from fastrtc.speech_to_text.stt_ import STTModel
from fastrtc.utils import audio_to_float32
from numpy.typing import NDArray

from funasr_utils.resample import resample_audio

stt_logger = logging.getLogger("STT")

from funasr import AutoModel  # noqa: E402


class LocalFunASR(STTModel):

    def __init__(self):
        self.model = None

        if torch.cuda.is_available():
            device = "cuda:0"
            stt_logger.info("CUDA device detected, using GPU acceleration")
        else:
            device = "cpu"
            stt_logger.info("No CUDA device detected, using CPU")

        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            hub="ms",
            device=device,
            disable_update=True,
        )

    def stt(self, audio: Tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        sample_rate, audio_array = audio
        stt_logger.info(
            f"Received audio data: sample_rate={sample_rate}, data_type={type(audio_array)}, shape={getattr(audio_array, 'shape', 'N/A')}"
        )

        if self.model is None:
            raise RuntimeError("ASR model not loaded, please initialize model first")

        try:
            audio_array = audio_to_float32(audio_array)
            sample_rate, audio_array = resample_audio(
                audio_array, sample_rate, stt_logger
            )

            if audio_array.ndim > 1:
                audio_array = (
                    audio_array[0]
                    if audio_array.shape[0] < audio_array.shape[1]
                    else audio_array[:, 0]
                )

            stt_logger.info(f"Processed audio length: {len(audio_array)}")

            result = self.model.generate(
                input=audio_array,
                cache={},
                language="zh",
                use_itn=True,
                batch_size_s=60,
                disable_pbar=True,
            )

            text = result[0]["text"] if result and len(result) > 0 else ""

            parsed_text = self._parse_funasr_output(text)

            return parsed_text if parsed_text else ""

        except Exception as e:
            stt_logger.error(f"Speech transcription error: {e}")
            return f"Transcription failed: {str(e)}"

    def _parse_funasr_output(self, raw_text: str) -> str:
        if not raw_text:
            return ""

        try:
            import re

            pattern = r"<\|([^|]+)\|><\|([^|]+)\|><\|([^|]+)\|><\|([^|]+)\|>(.*)"
            match = re.match(pattern, raw_text)

            if match:
                language = match.group(1)
                emotion = match.group(2)
                speech_type = match.group(3)
                processing = match.group(4)
                actual_text = match.group(5)

                stt_logger.debug(
                    f"Parse result - language: {language}, emotion: {emotion}, type: {speech_type}, processing: {processing}"
                )
                stt_logger.debug(f"Extracted text: {actual_text}")

                return actual_text.strip()
            else:
                stt_logger.warning(
                    f"Could not parse funasr output format, returning original text: {raw_text}"
                )
                return raw_text

        except Exception as e:
            stt_logger.error(f"Error parsing funasr output: {e}")
            return raw_text
