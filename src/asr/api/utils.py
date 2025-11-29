from typing import BinaryIO
import numpy as np

from env import envs
import ffmpeg
import imageio_ffmpeg

def load_audio(file: BinaryIO, encode=True, sr: int = envs.stt_sample_rate):
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
            # TODO: I can not get this ffmpeg work.
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # 使用配置的FFmpeg路径（本地优先）
            ffmpeg_cmd = imageio_ffmpeg.get_ffmpeg_exe()
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd=ffmpeg_cmd, capture_stdout=True, capture_stderr=True, input=file.read())
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = file.read()

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
