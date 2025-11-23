import os
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional
import json
import logging


class VoiceConfig(BaseModel):
    name: str
    description: str
    filename: str
    text: Optional[str] = None
    language: str

    def get_file_path(self, voice_dir: str) -> str:
        return os.path.join(voice_dir, self.filename)

    def get_text(self) -> str:
        if self.text is not None:
            return self.text
        raise ValueError(f"文本未设置")

    def get_language(self) -> str:
        return self.language

    def get_description(self) -> str:
        return self.description

    def get_voice_name(self) -> str:
        return self.name


class VoiceConfigMap(BaseModel):
    voices: Dict[str, VoiceConfig]  # 改为字典类型

    @field_validator("voices", mode="before")
    @classmethod
    def convert_list_to_dict(cls, v):
        if isinstance(v, list):
            return {voice["name"]: voice for voice in v}
        return v

    def get_voice_config(self, voice_type: str, default_voice: str) -> VoiceConfig:
        voice = self.voices.get(voice_type, None)
        if voice is None:
            logging.error(f"声音 {voice_type} 不存在，使用默认声音 {default_voice}")
            voice = self.voices.get(default_voice, None)
            if voice is None:
                raise ValueError(f"默认声音 {default_voice} 也不存在")
        return voice


def load_voice_config(config_path: str) -> VoiceConfigMap:
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return VoiceConfigMap(**data)
