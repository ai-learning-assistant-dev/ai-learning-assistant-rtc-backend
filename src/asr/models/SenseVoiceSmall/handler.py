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
        config_path: ASR model only config
        vad_path: VAD model path
        """
        import os
        
        with open(config_path) as f:
            config = toml.load(f)

        self.device = (
            "cuda" if torch.cuda.is_available() and envs.use_gpu else "cpu"
        )
        self.vad_path = vad_path
        self.model_name = config["model"]["name"]
        self.model_hub = config["model"]["hub"]
        
        # 检查本地模型目录是否存在
        current_dir = os.path.dirname(os.path.abspath(config_path))
        local_model_path = os.path.join(current_dir, config["download"]["path"])
        
        # 如果本地模型目录存在，使用本地路径；否则使用模型名从hub下载
        if os.path.exists(local_model_path) and os.path.isdir(local_model_path):
            self.model_path = local_model_path
            model_to_load = self.model_path
        else:
            self.model_path = None
            model_to_load = self.model_name

        self.model = AutoModel(
            model=model_to_load,
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

        if not result or len(result) == 0:
            return TranscribeData(text="", segments=[], language=language or "auto")

        # 提取文本和语言信息
        raw_text = result[0].get("text", "")
        detected_lang = self._extract_language(raw_text)
        parsed_text = self._parse_funasr_output(raw_text)

        # SenseVoice通常不返回时间戳信息，创建一个包含全文的单个片段
        # 如果未来FunASR版本支持时间戳，可以在这里添加处理逻辑
        segments = []
        if parsed_text:
            segments.append({
                "text": parsed_text,
                "start": 0,
                "end": 0,  # SenseVoice不提供时间戳
                "words": []  # SenseVoice不提供词级时间戳
            })

        return TranscribeData(
            text=parsed_text,
            segments=segments,
            language=detected_lang or language or "auto"
        )

    def language_detection(self, audio: np.ndarray) -> tuple[str, float]:
        """
        检测音频的语言
        
        Returns:
            tuple[str, float]: (语言代码, 置信度)
        """
        if self.model is None:
            raise RuntimeError("ASR model not loaded, please initialize model first")

        # 使用模型进行语言检测
        result = self.model.generate(
            input=audio,
            cache={},
            language="auto",  # 让模型自动检测语言
            use_itn=False,
            batch_size_s=60,
            merge_vad=True if self.vad_path is not None else False,
            disable_pbar=True,
        )

        if not result or len(result) == 0:
            return "unknown", 0.0

        raw_text = result[0].get("text", "")
        language = self._extract_language(raw_text)
        
        # SenseVoice不直接提供置信度，我们设置一个默认值
        # 如果提取到了语言标记，置信度设为0.9，否则为0.0
        confidence = 0.9 if language else 0.0

        return language or "unknown", confidence

    def _extract_language(self, raw_text: str) -> str:
        """从FunASR的输出中提取语言标记"""
        if not raw_text:
            return ""

        try:
            import re

            pattern = r"<\|([^|]+)\|>"
            match = re.match(pattern, raw_text)
            if match:
                return match.group(1)
        except Exception as e:
            logging.error(f"Error extracting language: {e}")
        
        return ""

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
            description="FunASR提供的ASR模型",
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
