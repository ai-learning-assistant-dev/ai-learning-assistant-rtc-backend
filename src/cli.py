import logging
import subprocess
import traceback
from typing import Annotated, Literal, Tuple

import torch
import typer
import uvicorn

from api import app
from asr.rtc_adapter import FSMNVad
from env import envs
from rtc.fastrtc_register import FastRTCRegister
from tts.models.model_manager import tts_model_manager

cli = typer.Typer()


def setup_logging(level):
    logging.basicConfig(
        level=level,
        format="[%(name)s]: %(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# 调用函数设置日志级别
setup_logging(logging.INFO)


def detect_cuda_environment():
    """检测CUDA环境是否可用"""
    try:
        # 检查PyTorch是否支持CUDA
        if torch.cuda.is_available():
            logging.info(f"检测到CUDA环境，GPU数量: {torch.cuda.device_count()}")
            return True
        else:
            logging.info("PyTorch未检测到CUDA环境")
            return False
    except Exception as e:
        logging.warning(f"CUDA检测失败: {e}")

    # 备用检测方法：使用nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            logging.info("通过nvidia-smi检测到CUDA环境")
            return True
        else:
            logging.info("nvidia-smi执行失败，判断为CPU环境")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        logging.info(f"nvidia-smi不可用: {e}")
        return False


def auto_select_model():
    """根据CUDA环境自动选择合适的TTS模型"""
    cuda_available = detect_cuda_environment()

    if cuda_available:
        selected_model = "index-tts"
        logging.info(f"CUDA环境可用，自动选择模型: {selected_model}")
    else:
        selected_model = "kokoro"
        logging.info(f"CUDA环境不可用，自动选择CPU模型: {selected_model}")

    return selected_model


def auto_select_rtc_model() -> Tuple[str, str]:
    """
    根据CUDA环境自动选择合适的RTC TTS模型
    目前暂时只支持kokoro模型
    """

    cuda_available = detect_cuda_environment()

    if cuda_available:
        selected_tts = "kokoro"
        selected_asr = "SenseVoiceSmall"
        logging.info(
            f"CUDA环境可用，自动选择模型: TTS {selected_tts}; ASR {selected_asr}"
        )
    else:
        selected_tts = "kokoro"
        selected_asr = "SenseVoiceSmall"
        logging.info(
            f"CUDA环境不可用，自动选择模型: TTS {selected_tts}; ASR {selected_asr}"
        )

    return selected_tts, selected_asr


@cli.command()
def download(
    model_names: Annotated[
        str, typer.Option(help="要下载的模型名称列表，用逗号分隔 (如 kokoro,f5-tts)")
    ],
):
    """下载模型及相关音色资源"""

    import tts.api.api_handler as _  # noqa: F401

    models = [name.strip() for name in model_names.split(",")]

    for model_name in models:
        try:
            path = tts_model_manager.download_model(model_name)
            typer.echo(f"成功下载模型: {model_name}，路径: {path}")
        except Exception as e:
            typer.echo(f"下载模型 {model_name} 失败: {str(e)}", err=True)
            logging.error(traceback.format_exc())  # 打印完整栈信息


@cli.command()
def run_tts(
    model_names: Annotated[
        str | None,
        typer.Option(
            help="要加载的模型名称列表，用逗号分隔(如 kokoro)。如果不指定，将自动检测CUDA环境选择模型"
        ),
    ] = None,
    port: Annotated[int, typer.Option(help="服务端口")] = 8000,
    auto_detect: Annotated[
        bool, typer.Option(help="自动检测CUDA环境并选择合适的模型")
    ] = False,
):
    """运行TTS服务命令，支持加载多个模型或自动选择模型"""

    import tts.api.api_handler as _  # noqa: F401

    # 如果启用了自动检测模式或未指定模型名称，则自动选择
    if auto_detect or not model_names:
        if model_names:
            typer.echo("同时指定了模型名称和自动检测，将优先使用自动检测")
        selected_model = auto_select_model()
        models = [selected_model]
        typer.echo(f"自动选择的模型: {selected_model}")
    else:
        models = [name.strip() for name in model_names.split(",")]
        typer.echo(f"手动指定的模型: {', '.join(models)}")

    for model_name in models:
        try:
            # model_manager could figure out whether the model is a RTC model
            tts_model_manager.load_model(model_name)
            typer.echo(f"成功加载模型: {model_name}")
        except Exception as e:
            typer.echo(f"加载模型 {model_name} 失败: {str(e)}", err=True)
            logging.error(traceback.format_exc())  # 打印完整栈信息
            return

    uvicorn.run(app, host="0.0.0.0", port=port)


@cli.command()
def run_rtc(
    base_url: Annotated[
        str, typer.Option(help="ASR/TTS语音后端的服务器URL。默认为localhost")
    ] = "localhost",
    asr_name: Annotated[
        str,
        typer.Option(
            help="要加载的ASR模型名称（仅限一个模型）。如果不指定，将自动检测CUDA环境选择模型。例如：SenseVoiceSmall。"
        ),
    ] = "SenseVoiceSmall",
    vad_name: Annotated[
        Literal["Silero", "FSMN"],
        typer.Option(
            help="要加载的VAD模型名称（仅限一个模型）。如果不指定，将使用默认VAD模型。例如：FSMN。"
        ),
    ] = "Silero",
    tts_name: Annotated[
        str,
        typer.Option(
            help="要加载的TTS模型名称（仅限一个模型）。如果不指定，将自动检测CUDA环境选择模型。例如：kokoro。"
        ),
    ] = "kokoro",
    port: Annotated[int, typer.Option(help="服务端口")] = envs.rtc_port,
    auto_detect: Annotated[
        bool, typer.Option(help="自动检测CUDA环境并选择合适的模型")
    ] = False,
):
    """运行实时语音(RTC)服务命令，使用WebRTC进行音频传输"""
    # 初始化完成fastrtc_register后再导入API处理器
    import rtc.api.api_handler as _  # noqa: F401

    global fastrtc_register

    if auto_detect:
        tts_name, asr_name = auto_select_rtc_model()

    try:
        vad_model = None
        if vad_name == "FSMN":
            vad_model = FSMNVad()

        fastrtc_register = FastRTCRegister(base_url, tts_name, asr_name, vad_model)
        fastrtc_register.stream.mount(app)
    except Exception as e:
        typer.echo(f"加载模型 {tts_name} 失败: {str(e)}", err=True)
        logging.error(traceback.format_exc())  # 打印完整栈信息
        return

    uvicorn.run(app, host=envs.app_host, port=port)


@cli.command()
def run_asr(
    asr_names: Annotated[
        str,
        typer.Option(
            help="要加载的ASR模型名称（仅限一个模型）。如果不指定，将使用默认模型SenseVoiceSmall"
        ),
    ] = envs.default_asr_model,
    port: Annotated[int, typer.Option(help="服务端口")] = envs.asr_port,
    auto_detect: Annotated[
        bool, typer.Option(help="自动检测CUDA环境并选择合适的模型")
    ] = False,
):
    """运行ASR服务命令，支持语音转文字功能"""

    import asr.api.api_handler as _  # noqa: F401
    from asr.models.model_manager import asr_model_manager

    model_name = asr_names.strip()

    typer.echo(f"使用模型: {model_name}")

    try:
        asr_model_manager.load_model(model_name)
        typer.echo(f"成功加载模型: {model_name}")
    except Exception as e:
        typer.echo(f"加载模型 {model_name} 失败: {str(e)}", err=True)
        logging.error(traceback.format_exc())
        return

    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    cli()
