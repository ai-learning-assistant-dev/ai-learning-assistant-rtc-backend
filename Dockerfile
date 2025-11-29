FROM ghcr.io/astral-sh/uv:debian-slim 

RUN apt-get update && apt-get install -y git ffmpeg && cd / && \
	git clone https://github.com/ai-learning-assistant-dev/ai-learning-assistant-rtc-backend && \
	cd /ai-learning-assistant-rtc-backend && git checkout stable && rm -rf .git && apt-get remove -y git && apt-get autoremove -y
# CPU 版本镜像
RUN cd /ai-learning-assistant-rtc-backend && uv sync --extra cpu && uv run model_cache.py

ENV LLM_STREAM_URL=http://ai-learning-assistant-training-server:3000/api/ai-chat/chat/stream
ENV APP_PORT=8989
ENV APP_HOST=0.0.0.0

EXPOSE 8989
WORKDIR /ai-learning-assistant-rtc-backend

CMD ["uv", "run", "--no-sync", "main.py"]
