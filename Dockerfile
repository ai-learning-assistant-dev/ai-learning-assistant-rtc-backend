FROM ghcr.io/astral-sh/uv:debian-slim 

RUN apt-get update && apt-get install -y git ffmpeg && cd / && \
	git clone https://github.com/ai-learning-assistant-dev/ai-learning-assistant-rtc-backend

# NVIDIA GPU 版本镜像
# RUN cd /ai-learning-assistant-rtc-backend && uv sync --extra cu128 && uv run model_cache.py

# CPU 版本镜像
RUN cd /ai-learning-assistant-rtc-backend && uv sync --extra cpu && uv run model_cache.py

ENV LLM_STREAM_URL=http://ai-learning-assistant-training-server:3000/api/ai-chat/chat/stream
ENV RTC_PORT=8989
ENV APP_HOST=0.0.0.0

EXPOSE 8989
WORKDIR /ai-learning-assistant-rtc-backend

CMD ["uv", "run", "main.py"]
