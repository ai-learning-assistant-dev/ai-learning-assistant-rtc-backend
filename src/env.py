from pydantic_settings import BaseSettings


class EnvVar(BaseSettings):
    llm_stream_url: str = "http://localhost:3000/api/ai-chat/chat/stream"
    rtc_port: int = 8989
    tts_port: int = 8000
    asr_port: int = 9000
    app_host: str = "0.0.0.0"
    tts_sample_rate: int = 16000
    default_tts_model: str = "kokoro"
    asr_sample_rate: int = 16000
    default_asr_model: str = "SenseVoiceSmall"
    use_gpu: bool = True

    class Config:
        env_file = ".env"


envs = EnvVar()
