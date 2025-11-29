# api/main.py
import io
import logging
import traceback
import typing
from typing import Callable, List, Union

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from api import app
from env import envs

from ..models.model_interface import AsyncTTSModelInterface, ModelDetail, TTSModelInterface
from ..models.model_manager import tts_model_manager
from .utils import is_text_too_complex, split_text_safely

speed_doc = """
语速控制参数：

- **常规用法**：指定固定的语速倍数
  - `0.5` = 慢速
  - `1.0` = 正常语速（默认值）
  - `1.5` = 快速
  - `2.0` = 倍速

- **特殊用法**：当设置为 `0` 时，启用智能动态语速功能
  - 系统会根据文本长度自动调整语速
  - 短文本使用正常语速
  - 中等长度文本稍慢
  - 长文本使用更慢语速以优化用户体验
"""

class TTSRequest(BaseModel):
    input: str
    voice: str
    response_format: str
    speed: float = Field(default=1.0, description=speed_doc)
    model: str

# text longer, read slower
def _speed_callable(len_ps: int) -> float:
    speed = 0.8
    if len_ps <= 83:
        speed = 1
    elif len_ps < 183:
        speed = 1 - (len_ps - 83) / 500
    return speed * 1.1

# original: /v1/audio/speech
@app.post("/v1/tts/synthesize")
async def tts_handler(request: TTSRequest):
    logging.info(f"收到TTS请求: {request}")
    try:
        model_name = envs.default_tts_model if not request.model else request.model
        model = tts_model_manager.get_model(model_name)
        if model is None:
            raise ValueError(f"模型{model_name}未加载")
        if request.speed == 0:
            speed = _speed_callable
        else:
            speed = request.speed
        # 处理音频生成
        if is_text_too_complex(request.input, model.max_input_length()):
            text_segments = split_text_safely(request.input, model.max_input_length())
            segment_audios = []
            for segment in text_segments:
                logging.info(f"正在处理文本片段: {segment}")
                audio_data = model.synthesize(segment, request.voice, speed)
                segment_audios.append(audio_data)
            combined_audio = np.concatenate(segment_audios)
        else:
            combined_audio = model.synthesize(
                request.input, request.voice, speed
            )

        # 创建内存中的音频文件
        audio_buffer = io.BytesIO()
        sample_rate = envs.tts_sample_rate
        sf.write(
            audio_buffer, combined_audio, sample_rate, format=request.response_format
        )
        audio_buffer.seek(0)

        return StreamingResponse(
            audio_buffer,
            media_type=f"audio/{request.response_format}",
            headers={
                "Content-Disposition": f"attachment; filename=audio.{request.response_format}"
            },
        )
    except Exception as e:
        logging.error(f"TTS处理出错: {str(e)}", exc_info=True)
        logging.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"TTS处理出错: {str(e)}")


class RTCTTSRequest(BaseModel):
    input: str
    voice: str = Field(default="", description="语音类型")
    speed: float = Field(
        default=1.0,
        ge=0.0,
        description=speed_doc,
    )
    model: str


@app.post("/v1/tts/synthesize/stream")
def stream_tts_handler(request: RTCTTSRequest):
    logging.info(f"收到RTC TTS请求: {request}")
    try:
        model_name = envs.default_tts_model if not request.model else request.model
        model = tts_model_manager.get_model(model_name)
        if model is None:
            raise ValueError(f"模型{model_name}未加载")
        model_info = model.get_model_info()
        if not model_info.is_rtc_model:
            raise ValueError("Can not use this TTS model for RTC")
        model = typing.cast(AsyncTTSModelInterface, model)
        sample_rate = model_info.sample_rate

        async def iter_stream():
            if request.speed == 0:
                speed = _speed_callable
            else:
                speed = request.speed
            # 流处理的情况下可以不用对文本做预处理，因为内部会完成文本切分
            async for audio_chunk in model.stream_synthesize(
                request.input, request.voice, speed
            ):
                if audio_chunk.dtype == np.float32 or audio_chunk.dtype == np.float64:
                    pcm_data = (audio_chunk * 32767).astype(np.int16)
                else:
                    pcm_data = audio_chunk
                yield pcm_data.tobytes()
            # 创建内存中的音频文件

        return StreamingResponse(
            iter_stream(),
            media_type="audio/pcm",
            headers={
                "Content-Disposition": "attachment; filename=audio.pcm",
                "X-Audio-Sample-Rate": str(sample_rate),
                "X-Audio-Channels": "1",  # 假设单声道
                "X-Audio-Bits": "16",  # 16位深度
                "X-Audio-Encoding": "signed-integer",  # 有符号整数
                "X-Audio-Format": "PCM",
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )
    except Exception as e:
        logging.error(f"TTS处理出错: {str(e)}", exc_info=True)
        logging.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"TTS处理出错: {str(e)}")


class AvailableModelsResponse(BaseModel):
    models: List[ModelDetail]


# original: /v1/models/info
@app.get("/v1/tts/models/info", response_model=AvailableModelsResponse)
async def get_available_models_info():
    """
    获取所有可用TTS模型及其音色信息。
    """
    all_model_details: List[ModelDetail] = []
    try:
        available_models = tts_model_manager.get_available_models()
        for model_name in available_models:
            model_instance = tts_model_manager.get_model(model_name)
            if model_instance is None:
                # This is impossible, but we still have this error handling
                raise ValueError(f"模型{model_name}未加载")
            model_info = model_instance.get_model_info()  # 获取模型基本信息
            all_model_details.append(model_info)
        return AvailableModelsResponse(models=all_model_details)
    except Exception as e:
        logging.error(f"获取模型信息时出错: {str(e)}", exc_info=True)
        logging.debug(traceback.format_exc())  # 记录完整的堆栈信息以便调试
        raise HTTPException(status_code=500, detail=f"获取模型信息时出错: {str(e)}")


@app.get("/v1/tts/models/info/stream", response_model=AvailableModelsResponse)
async def get_available_stream_models_info(model_name: str):
    """
    get all TTS models that has the ability to be streamingly used
    """
    all_model_details: List[ModelDetail] = []
    try:
        available_models = tts_model_manager.get_available_rtc_models()
        for model_name in available_models:
            model_instance = tts_model_manager.get_model(model_name)
            if model_instance is None:
                # This is impossible, but we still have this error handling
                raise ValueError(f"模型{model_name}未加载")
            model_info = model_instance.get_model_info()  # 获取模型基本信息
            all_model_details.append(model_info)
        return AvailableModelsResponse(models=all_model_details)
    except Exception as e:
        logging.error(f"获取模型信息时出错: {str(e)}", exc_info=True)
        logging.debug(traceback.format_exc())  # 记录完整的堆栈信息以便调试
        raise HTTPException(status_code=500, detail=f"获取模型信息时出错: {str(e)}")
