import json
from io import StringIO
from typing import BinaryIO

import ffmpeg
import imageio_ffmpeg
import numpy as np
from numpy import float32
from numpy.typing import NDArray

from asr.models.model_interface import TranscribeData
from env import envs


def load_audio(file: BinaryIO, encode=True, sr: int = envs.asr_sample_rate):
    """
    打开音频文件对象并读取为单声道波形，必要时重新采样。

    Parameters
    ----------
    file: BinaryIO
        音频文件对象
    encode: Boolean
        如果为True，通过FFmpeg编码音频流为WAV格式
    sr: int
        重新采样的目标采样率
    Returns
    -------
    包含音频波形的NumPy数组，float32类型。
    """
    if encode:
        try:
            # 使用 imageio_ffmpeg 提供的 ffmpeg（自动下载，无需本地安装）
            ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(
                    cmd=ffmpeg_cmd,
                    capture_stdout=True,
                    capture_stderr=True,
                    input=file.read(),
                )
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = file.read()

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def format_timestamp(milliseconds: int, format_type: str = "srt") -> str:
    """
    将毫秒时间戳格式化为指定格式

    Parameters
    ----------
    milliseconds: int
        毫秒时间戳
    format_type: str
        格式类型，支持 "srt" (00:00:00,000) 或 "vtt" (00:00:00.000)

    Returns
    -------
    格式化后的时间戳字符串
    """
    hours = milliseconds // 3600000
    minutes = (milliseconds % 3600000) // 60000
    seconds = (milliseconds % 60000) // 1000
    ms = milliseconds % 1000

    separator = "," if format_type == "srt" else "."
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{ms:03d}"


def convert_to_txt(result: TranscribeData) -> StringIO:
    """将识别结果转换为纯文本格式"""
    output = StringIO()
    output.write(result.text)
    output.seek(0)
    return output


def convert_to_json(result: TranscribeData) -> StringIO:
    """将识别结果转换为JSON格式"""
    output = StringIO()
    json.dump(result.model_dump(), output, ensure_ascii=False, indent=2)
    output.seek(0)
    return output


def convert_to_vtt(result: TranscribeData) -> StringIO:
    """
    将识别结果转换为WebVTT格式
    注意：SenseVoice不提供时间戳，时间戳将显示为 00:00:00.000
    """
    output = StringIO()
    output.write("WEBVTT\n\n")

    for i, segment in enumerate(result.segments, 1):
        start = format_timestamp(segment.get("start", 0), "vtt")
        end = format_timestamp(segment.get("end", 0), "vtt")
        text = segment.get("text", "")

        output.write(f"{i}\n")
        output.write(f"{start} --> {end}\n")
        output.write(f"{text}\n\n")

    output.seek(0)
    return output


def convert_to_srt(result: TranscribeData) -> StringIO:
    """
    将识别结果转换为SRT字幕格式
    注意：SenseVoice不提供时间戳，时间戳将显示为 00:00:00,000
    """
    output = StringIO()

    for i, segment in enumerate(result.segments, 1):
        start = format_timestamp(segment.get("start", 0), "srt")
        end = format_timestamp(segment.get("end", 0), "srt")
        text = segment.get("text", "")

        output.write(f"{i}\n")
        output.write(f"{start} --> {end}\n")
        output.write(f"{text}\n\n")

    output.seek(0)
    return output


def convert_to_tsv(result: TranscribeData) -> StringIO:
    """将识别结果转换为TSV（Tab Separated Values）格式"""
    output = StringIO()
    output.write("start\tend\ttext\n")

    for segment in result.segments:
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "")
        output.write(f"{start}\t{end}\t{text}\n")

    output.seek(0)
    return output


def convert_result_format(result: TranscribeData, output_format: str) -> StringIO:
    """
    将识别结果转换为指定的输出格式

    Parameters
    ----------
    result: TranscribeData
        识别结果
    output_format: str
        输出格式：txt, json, vtt, srt, tsv

    Returns
    -------
    StringIO对象，包含格式化后的结果
    """
    format_converters = {
        "txt": convert_to_txt,
        "json": convert_to_json,
        "vtt": convert_to_vtt,
        "srt": convert_to_srt,
        "tsv": convert_to_tsv,
    }

    converter = format_converters.get(output_format)
    if converter is None:
        raise ValueError(f"不支持的输出格式: {output_format}")

    return converter(result)


def resample_audio(
    audio_array: NDArray[float32],
    source_sample_rate: int,
    target_sample_rate: int,
    logger=None,
) -> tuple[int, NDArray[float32]]:
    """
    音频重采样

    Args:
        audio_array: 原始音频数据（要求NDArray[float32]）
        sample_rate: 原始采样率
        target_sample_rate: 期望采样率

    Returns:
        重采样后的音频数据
    """

    if source_sample_rate == target_sample_rate:
        return source_sample_rate, audio_array

    if logger:
        logger.info(f"Resampling from {source_sample_rate}Hz to {target_sample_rate}Hz")
    try:
        import librosa

        resampled_audio = librosa.resample(
            audio_array.astype(np.float32),
            orig_sr=source_sample_rate,
            target_sr=target_sample_rate,
        )
        return target_sample_rate, resampled_audio
    except ImportError:
        if logger:
            logger.warning("librosa not available, please try this: `uv add librosa`")
            # 如果重采样失败，返回原始数据并记录警告
            logger.warning(
                "Resampling failed, using original audio (may affect ASR accuracy)"
            )
        return source_sample_rate, audio_array
    except Exception as e:
        if logger:
            logger.error(f"Audio resampling error: {e}")
            logger.warning(
                "Resampling failed, using original audio (may affect ASR accuracy)"
            )
        return source_sample_rate, audio_array
