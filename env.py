from pydantic_settings import BaseSettings


class EnvVar(BaseSettings):
    llm_stream_url: str = "http://localhost:3000/api/ai-chat/chat/stream"
    app_port: int = 8989
    app_host: str = "0.0.0.0"
    audio_sample_rate: int = 16000
    default_stt_model: str = "kokoro"
    use_gpu: bool = True

    class Config:
        env_file = ".env"


envs = EnvVar()