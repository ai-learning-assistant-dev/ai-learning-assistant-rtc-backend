import numpy as np
from numpy.typing import NDArray
from numpy import float32


def resample_audio(
    audio_array: NDArray[float32], sample_rate: int, logger=None
) -> tuple[int, NDArray[float32]]:
    """
    音频重采样

    Args:
        audio_array: 原始音频数据（要求NDArray[float32]）
        sample_rate: 原始采样率

    Returns:
        16000Hz重采样后的音频数据
    """
    target_sample_rate = 16000  # FunASR内部使用的采样频率

    if sample_rate == target_sample_rate:
        return sample_rate, audio_array

    if logger:
        logger.info(f"Resampling from {sample_rate}Hz to {target_sample_rate}Hz")
    try:
        import librosa

        resampled_audio = librosa.resample(
            audio_array.astype(np.float32),
            orig_sr=sample_rate,
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
        return sample_rate, audio_array
    except Exception as e:
        if logger:
            logger.error(f"Audio resampling error: {e}")
            logger.warning(
                "Resampling failed, using original audio (may affect ASR accuracy)"
            )
        return sample_rate, audio_array
