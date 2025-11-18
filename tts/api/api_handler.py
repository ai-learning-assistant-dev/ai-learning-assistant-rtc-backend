# api/main.py
import io
import logging
import traceback
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from env import envs
from ..models.model_manager import model_manager
from .utils import is_text_too_complex, split_text_safely
import soundfile as sf
from ..models.model_interface import ModelDetail

from api import app


class TTSRequest(BaseModel):
    input: str
    voice: str
    response_format: str
    speed: float = 1.0
    model: str


@app.post("/v1/audio/speech")
async def tts_handler(request: TTSRequest):
    logging.info(f"收到TTS请求: {request}")
    try:
        model_name = envs.default_stt_model if not request.model else request.model
        model = model_manager.get_model(model_name)
        # 处理音频生成
        if is_text_too_complex(request.input, model.max_input_length()):
            text_segments = split_text_safely(request.input, model.max_input_length())
            segment_audios = []
            for segment in text_segments:
                logging.info(f"正在处理文本片段: {segment}")
                audio_data = model.synthesize(segment, request.voice, request.speed)
                segment_audios.append(audio_data)
            combined_audio = np.concatenate(segment_audios)
        else:
            combined_audio = model.synthesize(
                request.input, request.voice, request.speed
            )

        # 创建内存中的音频文件
        audio_buffer = io.BytesIO()
        sample_rate = int(envs.audio_sample_rate)
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


class AvailableModelsResponse(BaseModel):
    models: List[ModelDetail]


@app.get("/v1/models/info", response_model=AvailableModelsResponse)
async def get_available_models_info():
    """
    获取所有可用TTS模型及其音色信息。
    """
    all_model_details: List[ModelDetail] = []
    try:
        available_models = model_manager.get_available_models()
        for model_name in available_models:
            model_instance = model_manager.get_model(model_name)
            model_info = model_instance.get_model_info()  # 获取模型基本信息
            all_model_details.append(model_info)
        return AvailableModelsResponse(models=all_model_details)
    except Exception as e:
        logging.error(f"获取模型信息时出错: {str(e)}", exc_info=True)
        logging.debug(traceback.format_exc())  # 记录完整的堆栈信息以便调试
        raise HTTPException(status_code=500, detail=f"获取模型信息时出错: {str(e)}")
