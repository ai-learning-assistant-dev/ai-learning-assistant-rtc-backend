from importlib.resources import files
import os
from indextts.infer import IndexTTS
from huggingface_hub import hf_hub_download
import numpy as np
import torch
from typing import Union, Callable

from ..model_interface import TTSModelInterface, ModelDetail, VoiceDetail
import toml
from .. import voice_util
from env import envs


class TTSModel(TTSModelInterface):
    def __init__(self, config_path: str):
        with open(config_path, encoding="utf-8") as f:
            config = toml.load(f)

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.default_voice = config["voice"]["default_voice"]
        self.model_name = config["model"]["name"]
        self.model_path = config["paths"]["model_path"]
        self.model_path = os.path.join(self.current_dir, self.model_path)
        self.voice_dir = os.path.join(self.current_dir, config["paths"]["voice_path"])
        self.available_voices = voice_util.load_voice_config(
            os.path.join(self.voice_dir, "voice_config.json")
        )

        self.use_fast_infer = config["performance"]["use_fast_infer"]
        use_gpu = envs.use_gpu == "true" and torch.cuda.is_available()
        device = None
        if not use_gpu:
            device = "cpu"

        cfg_path = os.path.join(self.model_path, "config.yaml")
        self.tts = IndexTTS(model_dir=self.model_path, cfg_path=cfg_path, device=device)

    def synthesize(
        self,
        text: str,
        voice_type: str,
        _: Union[float, Callable[[int], float]] = 1,    # speed is ignored in index-tts
    ) -> np.ndarray:
        voice = self.available_voices.get_voice_config(voice_type, self.default_voice)
        if self.use_fast_infer:
            _, wav = self.tts.infer_fast(
                text=text,
                audio_prompt=voice.get_file_path(self.voice_dir),
                output_path=None,
            )
        else:
            _, wav = self.tts.infer(
                text=text,
                audio_prompt=voice.get_file_path(self.voice_dir),
                output_path=None,
            )
        return wav

    def get_model_info(self) -> ModelDetail:
        return ModelDetail(
            model_name=self.model_name,
            device=self.tts.device,
            voices=[
                VoiceDetail(name=voice.name, description=voice.description)
                for voice in self.available_voices.voices.values()
            ],
            description="IndexTTS 模型效果最优。支持音色克隆。建议在GPU环境使用。",
            max_input_length=self.max_input_length(),
            sample_rate=24000,  # IndexTTS 固定采样率为 24000
            is_rtc_model=False, # IndexTTS can not have streaming output
        )

    def max_input_length(self) -> int:
        return 300

    @staticmethod
    def download_model() -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        with open(
            os.path.join(current_dir, "model_config.toml"), encoding="utf-8"
        ) as f:
            config = toml.load(f)

        model_path = config["paths"]["model_path"]
        model_path = os.path.join(current_dir, model_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        repo_id = "IndexTeam/IndexTTS-1.5"
        filenames = [
            "config.yaml",
            "bigvgan_discriminator.pth",
            "bigvgan_generator.pth",
            "bpe.model",
            "dvae.pth",
            "gpt.pth",
            "unigram_12000.vocab",
        ]
        for filename in filenames:
            hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path)
        return model_path

    @staticmethod
    def create() -> "TTSModel":
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config = os.path.join(current_dir, "model_config.toml")
        return TTSModel(config)
