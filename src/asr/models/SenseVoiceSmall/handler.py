import logging

import numpy as np
import toml
import torch
from funasr import AutoModel
from modelscope.hub.snapshot_download import snapshot_download

from asr.models.model_interface import ASRModelInterface, ModelDetail, TranscribeData
from env import envs


class ASRModel(ASRModelInterface):
    def __init__(self, config_path: str, vad_path: str | None):
        """
        config_path: STT model only config
        vad_path: VAD model path
        """
        with open(config_path) as f:
            config = toml.load(f)

        self.device = (
            "cuda" if torch.cuda.is_available() and envs.use_gpu == "true" else "cpu"
        )
        self.vad_path = vad_path
        self.model_path = config["model"]["path"]
        self.model_name = config["model"]["name"]
        self.model_hub = config["model"]["hub"]
        self.model_path = config["download"]["path"]

        self.model = AutoModel(
            model=self.model_name if self.model_path is None else self.model_path,
            vad_model=self.vad_path,
            vad_kwargs={"max_single_segment_time": 30000},
            hub=self.model_hub,
            device=self.device,
            disable_update=False,  # model space, we could update it
        )

    def raw_transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        """
        This function assumes that the sample_rate of audio is correct.
        You should be careful about the input audio sample_rate.
        Return value: the transcribed text string
        """
        if self.model is None:
            raise RuntimeError("ASR model not loaded, please initialize model first")

        result = self.model.generate(
            input=audio,
            cache={},
            language=language if language is not None else "auto",
            use_itn=True,  # get the result with punctuation
            batch_size_s=60,
            merge_vad=True if self.vad_path is not None else False,
            disable_pbar=True,
        )

        text = result[0]["text"] if result and len(result) > 0 else ""

        parsed_text = self._parse_funasr_output(text)
        return parsed_text

    def transcribe(
        self, audio: np.ndarray, language: str | None = None
    ) -> TranscribeData:
        # TODO: need to merge the original logic of ASR
        return TranscribeData(text="", segments=[], language="")

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

                logging.debug(
                    f"Parse result - language: {language}, emotion: {emotion}, type: {speech_type}, processing: {processing}"
                )
                logging.debug(f"Extracted text: {actual_text}")

                return actual_text.strip()
            else:
                logging.warning(
                    f"Could not parse funasr output format, returning original text: {raw_text}"
                )
                return raw_text

        except Exception as e:
            logging.error(f"Error parsing funasr output: {e}")
            return raw_text

    def get_model_info(self) -> ModelDetail:
        return ModelDetail(
            model_name=self.model_name,
            device=self.device,
            description="FunASR提供的STT模型",
            sample_rate=16000,
        )

    @staticmethod
    def download_model() -> str:
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config = os.path.join(current_dir, "model_config.toml")
        with open(config) as f:
            config = toml.load(f)
        model_id = config["model"]["id"]
        download_path = config["download"]["path"]
        os.makedirs(download_path, exist_ok=True)
        logging.info(f"Downloading {model_id}")
        return snapshot_download(model_id, local_dir=download_path)

    @staticmethod
    def create(vad_model: str | None = None) -> "ASRModelInterface":
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config = os.path.join(current_dir, "model_config.toml")
        return ASRModel(config, vad_model)
