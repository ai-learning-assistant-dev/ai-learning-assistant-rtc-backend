import logging
import traceback
import typing
from typing import List
from urllib.parse import quote

import numpy as np
from fastapi import HTTPException, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from numpy.typing import NDArray
from pydantic import BaseModel, Field, field_validator

from api import app
from asr.api.utils import convert_result_format, load_audio, resample_audio
from asr.models.model_interface import ASRModelInterface, ModelDetail
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
            "Asr-Engine": (
                envs.default_asr_model
                if request.model_name is None
                else request.model_name
            ),
            "Content-Disposition": f'attachment; filename="{quote(request.audio_file.filename if request.audio_file.filename is not None else '')}.{request.output}"',
        },
    )


class RTCASRRequest(BaseModel):
    """RTC ASR 转录请求参数"""

    audio: NDArray[np.float32] = Field(
        ..., description="音频内容，需要确保采样率匹配模型要求"
    )
    sample_rate: int = Field(..., description="传入音频的采样率")
    language: str | None = Field(default=None, description="语言代码，不指定时自动检测")
    model_name: str | None = Field(
        default=None, description="ASR模型名称，不指定时使用默认模型"
    )

    @field_validator("audio")
    def validate_audio(cls, v):
        """验证音频数据"""
        if not v:
            raise ValueError("音频数据不能为空")
        if len(v) == 0:
            raise ValueError("音频数据长度不能为0")
        # 可选：验证音频数据范围
        # for sample in v:
        #     if not -1.0 <= sample <= 1.0:
        #         raise ValueError("音频样本值应在-1.0到1.0之间")
        return v

    class Config:
        arbitrary_types_allowed = True


@app.post("/v1/asr/transcribe/rtc", tags=["Endpoints"])
async def rtc_asr(request: RTCASRRequest):
    model = get_asr_model(request.model_name)
    model_info = model.get_model_info()
    if model_info.sample_rate != request.sample_rate:
        resample_audio(request.audio, request.sample_rate, model_info.sample_rate)

    result = model.raw_transcribe(
        request.audio,
        request.language,
    )

    return PlainTextResponse(content=result)


class AvailableModelsResponse(BaseModel):
    models: List[ModelDetail]


@app.get("/v1/asr/models/info", response_model=AvailableModelsResponse)
async def get_available_models_info():
    """
    获取所有可用ASR模型及其音色信息。
    """
    all_model_details: List[ModelDetail] = []
    try:
        available_models = asr_model_manager.get_available_models()
        for model_name in available_models:
            model_instance = typing.cast(
                ASRModelInterface,
                asr_model_manager.get_model(model_name),
            )
            model_info = model_instance.get_model_info()  # 获取模型基本信息
            all_model_details.append(model_info)
        return AvailableModelsResponse(models=all_model_details)
    except Exception as e:
        logging.error(f"获取模型信息时出错: {str(e)}", exc_info=True)
        logging.debug(traceback.format_exc())  # 记录完整的堆栈信息以便调试
        raise HTTPException(status_code=500, detail=f"获取模型信息时出错: {str(e)}")


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

    detected_lang, confidence = model.language_detection(
        load_audio(request.audio_file.file, request.encode)
    )

    return {
        "detected_language": detected_lang,
        "confidence": confidence,
    }
