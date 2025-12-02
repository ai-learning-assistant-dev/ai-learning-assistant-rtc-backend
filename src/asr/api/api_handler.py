import typing
from typing import Optional
from urllib.parse import quote

import click
import uvicorn
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api import app
from asr.api.utils import convert_result_format, load_audio
from asr.models.model_interface import ASRModelInterface
from env import envs

from ..models.model_manager import asr_model_manager


def get_asr_model(model_name: str | None = None):
    """获取ASR模型，加载SenseVoice模型"""
    if model_name is None:
        model_name = envs.default_asr_model

    if asr_model_manager.get_model(model_name) is None:
        print("正在加载ASR模型...")
        asr_model_manager.load_model(model_name)
        print(f"✓ ASR模型加载完成: {model_name}")
    asr_model = typing.cast(ASRModelInterface, asr_model_manager.get_model(model_name))
    return asr_model


class ASRRequest(BaseModel):
    """ASR转录请求参数"""

    audio_file: UploadFile = Field(..., description="音频文件上传")
    encode: bool = Field(default=True, description="是否通过ffmpeg编码音频")
    language: str | None = Field(default=None, description="语言代码，不指定时自动检测")
    output: str = Field(
        default="txt", description="输出格式", pattern="^(txt|vtt|srt|tsv|json)$"
    )
    model_name: str | None = Field(
        default=None, description="ASR模型名称，不指定时使用默认模型"
    )

    class Config:
        arbitrary_types_allowed = True


# original: /asr
@app.post("/v1/asr/transcribe", tags=["Endpoints"])
async def asr(request: ASRRequest):
    """
    语音转文字接口。

    参数说明：
    - language: 语言代码（如：zh, en, ja等），不指定时自动检测
    - output: 输出格式（txt, json, vtt, srt, tsv）

    注意：SenseVoice模型不提供时间戳信息，vtt/srt格式中时间戳将为0
    """
    model = get_asr_model(request.model_name)

    result = model.transcribe(
        load_audio(request.audio_file.file, request.encode),
        request.language,
    )

    formatted_result = convert_result_format(result, request.output)

    # 根据输出格式设置正确的媒体类型
    media_type_map = {
        "txt": "text/plain",
        "json": "application/json",
        "vtt": "text/vtt",
        "srt": "application/x-subrip",
        "tsv": "text/tab-separated-values",
    }

    return StreamingResponse(
        formatted_result,
        media_type=media_type_map.get(request.output, "text/plain"),
        headers={
            "Asr-Engine": envs.default_asr_model,
            "Content-Disposition": f'attachment; filename="{quote(request.audio_file.filename if request.audio_file.filename is not None else '')}.{request.output}"',
        },
    )


class DetectLanguageRequest(BaseModel):
    """ASR语言识别请求参数"""

    audio_file: UploadFile = Field(..., description="音频文件上传")
    encode: bool = Field(default=True, description="是否通过ffmpeg编码音频")
    model_name: str | None = Field(
        default=None, description="ASR模型名称，不指定时使用默认模型"
    )

    class Config:
        arbitrary_types_allowed = True


@app.post("/v1/asr/detect-language", tags=["Endpoints"])
async def detect_language(request: DetectLanguageRequest):
    """
    语言检测接口（不进行完整转录，节省计算资源）。

    使用场景：
    - 只需要检测语言，不需要转录内容
    - 前端根据检测到的语言做不同的处理逻辑

    注意：如果需要转录，建议直接使用 /v1/asr/transcribe 接口，它会自动包含语言信息。
    """
    model = get_asr_model(request.model_name)
    if model is None:
        raise ValueError("ASR模型未加载")

    detected_lang_code, confidence = model.language_detection(
        load_audio(request.audio_file.file, request.encode)
    )

    # 语言代码映射表
    LANGUAGE_NAMES = {
        "zh": "Chinese",
        "en": "English",
        "ja": "Japanese",
        "ko": "Korean",
        "yue": "Cantonese",
        "auto": "Auto-detect",
        "unknown": "Unknown",
    }

    return {
        "detected_language": LANGUAGE_NAMES.get(detected_lang_code, "Unknown"),
        "language_code": detected_lang_code,
        "confidence": confidence,
    }


@click.command()
@click.option(
    "-h",
    "--host",
    metavar="HOST",
    default="0.0.0.0",
    help="Host for the webservice (default: 0.0.0.0)",
)
@click.option(
    "-p",
    "--port",
    metavar="PORT",
    default=9000,
    help="Port for the webservice (default: 9000)",
)
# TODO: I removed the default metadata, if you find it useful, please add it back as you want.
# @click.version_option(version=app_version)  # 使用我们定义的版本变量
def start(host: str, port: Optional[int] = None):
    # 在启动服务前预加载模型
    print("正在启动ASR Webserver...")
    get_asr_model()
    print(f"服务启动在 http://{host}:{port}")
    if port is not None:
        uvicorn.run(app, host=host, port=port)
    else:
        uvicorn.run(app, host=host)


if __name__ == "__main__":
    start()
