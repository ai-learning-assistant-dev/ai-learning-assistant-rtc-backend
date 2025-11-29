from io import StringIO
from typing import Annotated, Optional, Union, cast
from urllib.parse import quote

import click
import uvicorn
from fastapi import File, Query, UploadFile
from fastapi.responses import StreamingResponse

from api import app
from asr.api.utils import load_audio

from ..models.model_manager import asr_model_manager


def get_asr_model(model_name: str):
    """获取ASR模型，加载"""
    if asr_model_manager.get_model(model_name) is None:
        print("正在加载ASR模型...")
        asr_model_manager.load_model(model_name)
        print(f"✓ ASR模型加载完成: {model_name}")
    asr_model = asr_model_manager.get_model(model_name)
    return asr_model


# original: /asr
@app.post("/v1/asr/transcribe", tags=["Endpoints"])
async def asr(
    # TODO: Think about how to add model_name here.
    # Because there might be multiple models available.
    # We might mimic the TTS logic: define a request type here.
    # But this will break the API, old plugins rely on this should change there codes.
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(
        default="transcribe", enum=["transcribe", "translate"]
    ),
    # TODO: Every model might has its own language codes, so enum might not works well here.
    language: Union[str, None] = Query(default=None),   #, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    vad_filter: Annotated[
        Optional[bool],
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=False,  # SenseVoice不支持VAD配置
        ),
    ] = False,
    word_timestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=False,  # SenseVoice不支持词级时间戳
    ),
    # TODO: We should process the output format in this function, not inside the model.
    output: Union[str, None] = Query(
        default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]
    ),
):
    # TODO: Same above, we need to know the model_name, but not from environment variables
    model_name = "SenseVoiceSmall"
    model = get_asr_model(model_name)
    if model is None:
        raise ValueError(f"模型{model_name}未加载")
    result = model.transcribe(
        load_audio(audio_file.file, encode),
        language,
    )
    # TODO: convert the result into StringIO
    result = cast(StringIO, result)  # this is wrong, remember to delete this

    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": model_name,
            "Content-Disposition": f'attachment; filename="{quote(audio_file.filename if audio_file.filename is not None else '')}.{output}"',
        },
    )


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    # TODO: Same above, we need to know the model_name, but not from environment variables
    model_name = "SenseVoiceSmall"
    model = get_asr_model(model_name)
    # TODO: We don't have language_detection interface yet.
    detected_lang_code, confidence = model.language_detection(
        load_audio(audio_file.file, encode)
    )
    return {
        # TODO: We might not using SENSEVOICE here, we should return the correct format inside the model
        "detected_language": SENSEVOICE_LANGUAGES.get(detected_lang_code, "Unknown"),
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
    print("正在启动VoiceWebserver...")
    # TODO: Same above, we need to know the model_name, but not from environment variables
    model_name = "SenseVoiceSmall"
    get_asr_model(model_name)
    print(f"服务启动在 http://{host}:{port}")
    if port is not None:
        uvicorn.run(app, host=host, port=port)
    else:
        uvicorn.run(app, host=host)


if __name__ == "__main__":
    start()
