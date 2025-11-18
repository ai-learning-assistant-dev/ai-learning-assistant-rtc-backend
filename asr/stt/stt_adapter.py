import logging
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import torch
from fastrtc.speech_to_text.stt_ import STTModel
from fastrtc.utils import audio_to_float32

from ..funasr_utils.resample import resample_audio

logging.basicConfig(
    level=logging.INFO,
    format="[STT]: %(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from funasr import AutoModel


class LocalFunASR(STTModel):

    def __init__(self):
        self.model = None

        if torch.cuda.is_available():
            device = "cuda:0"
            logger.info("CUDA device detected, using GPU acceleration")
        else:
            device = "cpu"
            logger.info("No CUDA device detected, using CPU")

        self.model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_kwargs={"max_single_segment_time": 30000},
            hub="ms",
            device=device,
            disable_update=True,
        )

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        sample_rate, audio_array = audio
        logger.info(
            f"Received audio data: sample_rate={sample_rate}, data_type={type(audio_array)}, shape={getattr(audio_array, 'shape', 'N/A')}"
        )

        if self.model is None:
            raise RuntimeError("ASR model not loaded, please initialize model first")

        try:
            audio_array = audio_to_float32(audio_array)
            sample_rate, audio_array = resample_audio(audio_array, sample_rate, logger)

            if audio_array.ndim > 1:
                audio_array = (
                    audio_array[0]
                    if audio_array.shape[0] < audio_array.shape[1]
                    else audio_array[:, 0]
                )

            logger.info(f"Processed audio length: {len(audio_array)}")

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
            logger.error(f"Speech transcription error: {e}")
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

                logger.debug(
                    f"Parse result - language: {language}, emotion: {emotion}, type: {speech_type}, processing: {processing}"
                )
                logger.debug(f"Extracted text: {actual_text}")

                return actual_text.strip()
            else:
                logger.warning(
                    f"Could not parse funasr output format, returning original text: {raw_text}"
                )
                return raw_text

        except Exception as e:
            logger.error(f"Error parsing funasr output: {e}")
            return raw_text
