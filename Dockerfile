FROM ghcr.io/astral-sh/uv:debian-slim AS download-code

RUN apt-get update
RUN apt-get install -y git
RUN cd / && git clone https://github.com/ai-learning-assistant-dev/ai-learning-assistant-rtc-backend
WORKDIR /ai-learning-assistant-rtc-backend
RUN git checkout stable && rm -rf .git

FROM ghcr.io/astral-sh/uv:debian-slim
RUN apt-get update
RUN apt-get install -y ffmpeg
COPY SenseVoiceSmall /root/.cache/modelscope/hub/models/iic/SenseVoiceSmall
WORKDIR /ai-learning-assistant-rtc-backend
COPY model_cache.py model_cache.py
COPY pyproject.toml pyproject.toml
# CPU 版本镜像
RUN uv sync --extra cpu
RUN uv run model_cache.py

COPY --from=download-code /ai-learning-assistant-rtc-backend /ai-learning-assistant-rtc-backend

ENV LLM_STREAM_URL=http://host.ala.internal:7100/api/ai-chat/chat/stream
ENV APP_PORT=8989
ENV APP_HOST=0.0.0.0

EXPOSE 8989

CMD ["uv", "run", "--no-sync", "main.py"]
