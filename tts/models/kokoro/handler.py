import logging
import os
from huggingface_hub import hf_hub_download
import torch
from kokoro import KModel, KPipeline
import numpy as np
import toml

from env import envs
from .. import voice_util
from ..model_interface import TTSModelInterface, ModelDetail, VoiceDetail


class TTSModel(TTSModelInterface):
    def __init__(self, config_path: str):
        with open(config_path) as f:
            config = toml.load(f)

        self.device = (
            "cuda" if torch.cuda.is_available() and envs.use_gpu == "true" else "cpu"
        )
        logging.info(f"使用设备: {self.device}")
        self.repo_id = config["model"]["repo_id"]
        self.current_dir = os.path.dirname(os.path.abspath(__file__))

        self.model_path = config["paths"]["model_path"]
        self.model_path = os.path.join(self.current_dir, self.model_path)
        self.config_path = os.path.join(
            self.current_dir, config["paths"]["config_path"]
        )

        en_pipeline = KPipeline(lang_code="a", repo_id=self.repo_id, model=False)

        def en_callable(text):
            return next(en_pipeline(text)).phonemes

        self.model = (
            KModel(repo_id=self.repo_id, model=self.model_path, config=self.config_path)
            .to(self.device)
            .eval()
        )
        self.pipeline = KPipeline(
            lang_code="z",
            repo_id=self.repo_id,
            model=self.model,
            en_callable=en_callable,
        )

        self.default_voice = config["voice"]["default_voice"]
        self.model_name = config["model"]["name"]
        voice_dir_name = config["paths"]["voice_path"]
        self.voice_dir = os.path.join(self.current_dir, voice_dir_name)
        self.available_voices = voice_util.load_voice_config(
            os.path.join(self.voice_dir, "voice_config.json")
        )

    def synthesize(self, text: str, voice_type: str, speed: float) -> np.ndarray:
        voice = self.available_voices.get_voice_config(voice_type, self.default_voice)
        voice_file_path = os.path.join(self.voice_dir, voice.filename)
        if not os.path.exists(voice_file_path):
            raise ValueError(f"音色文件不存在: {voice_file_path}")

        generator = self.pipeline(text, voice_file_path, speed)
        result_wav = None
        for result in generator:
            if result.audio is not None:
                wav = result.audio.cpu().numpy()
                result_wav = (
                    np.concatenate([result_wav, wav]) if result_wav is not None else wav
                )
        if result_wav is None:
            raise ValueError("合成的音频数据为空")
        return result_wav

    def get_model_info(self) -> ModelDetail:
        return ModelDetail(
            model_name=self.model_name,
            device=self.device,
            voices=[
                VoiceDetail(name=voice.name, description=voice.description)
                for voice in self.available_voices.voices.values()
            ],
            description="Kokoro 模型推理速度快",
            max_input_length=self.max_input_length(),
        )

    def max_input_length(self) -> int:
        return 100

    @staticmethod
    def download_model() -> str:
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config = os.path.join(current_dir, "model_config.toml")
        with open(config) as f:
            config = toml.load(f)
        repo_id = config["model"]["repo_id"]
        model_download_dir = os.path.join(current_dir, "model_download")
        os.makedirs(model_download_dir, exist_ok=True)
        TTSModel.download_voices()
        hf_hub_download(
            repo_id=repo_id, filename="config.json", local_dir=model_download_dir
        )
        return hf_hub_download(
            repo_id=repo_id,
            filename=KModel.MODEL_NAMES[repo_id],
            local_dir=model_download_dir,
        )

    @staticmethod
    def download_voices():
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config = os.path.join(current_dir, "model_config.toml")
        with open(config) as f:
            config = toml.load(f)
        repo_id = config["model"]["repo_id"]
        voice_dir_name = config["paths"]["voice_path"]
        voice_dir = os.path.join(current_dir, voice_dir_name)
        available_voices = voice_util.load_voice_config(
            os.path.join(voice_dir, "voice_config.json")
        )
        for voice in available_voices.voices.values():
            path = hf_hub_download(
                repo_id=repo_id,
                filename=f"voices/{voice.filename}",
                local_dir=voice_dir,
            )
            # 移动文件到目标位置
            target_path = os.path.join(voice_dir, voice.filename)
            if path != target_path:  # 如果路径不同才移动
                import shutil

                shutil.move(path, target_path)
                path = target_path
                # 删除空的voices文件夹
                voices_dir = os.path.join(voice_dir, "voices")
                if os.path.exists(voices_dir) and not os.listdir(voices_dir):
                    os.rmdir(voices_dir)
            logging.info(f"下载语音{voice.name}成功, 保存路径: {path}")

    @staticmethod
    def create() -> "TTSModel":
        import os

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config = os.path.join(current_dir, "model_config.toml")
        return TTSModel(config)
