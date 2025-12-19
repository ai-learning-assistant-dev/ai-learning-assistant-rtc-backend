from os import path

from fastrtc.pause_detection import SileroVADModel


class SileroVADModelModelScope(SileroVADModel):
    @staticmethod
    def download_model() -> str:
        return path.join(
            path.dirname(path.abspath(__file__)),
            "silero_vad.onnx",
        )

    # def __init__(self):
    #     try:
    #         import importlib.util
    #
    #         mod = importlib.util.find_spec("onnxruntime")
    #         if mod is None:
    #             raise RuntimeError("Install fastrtc[vad] to use ReplyOnPause")
    #     except (ValueError, ModuleNotFoundError):
    #         raise RuntimeError("Install fastrtc[vad] to use ReplyOnPause")
    #     super().__init__()
