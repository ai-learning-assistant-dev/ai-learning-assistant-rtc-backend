FROM ghcr.io/astral-sh/uv:debian-slim
RUN apt-get update
RUN apt-get install -y ffmpeg
WORKDIR /ai-learning-assistant-rtc-backend
COPY pyproject.toml pyproject.toml
# CPU 版本镜像
RUN uv sync --extra cpu
COPY models /root/.cache/modelscope/hub/models
COPY main.py main.py
COPY funasr_stt funasr_stt
COPY funasr_utils funasr_utils
COPY funasr_vad funasr_vad
COPY kokoro_tts kokoro_tts
COPY model_cache.py model_cache.py
ENV LLM_STREAM_URL=http://host.ala.internal:7100/api/ai-chat/chat/stream
ENV APP_PORT=8989
ENV APP_HOST=0.0.0.0

RUN uv run model_cache.py

EXPOSE 8989

CMD ["uv", "run", "--no-sync", "main.py"]
