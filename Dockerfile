FROM ghcr.io/astral-sh/uv:debian-slim 

RUN apt update && apt-get install -y git && cd / && \
	git clone https://github.com/ai-learning-assistant-dev/ai-learning-assistant-rtc-backend
# CPU 版本镜像
RUN cd /ai-learning-assistant-rtc-backend && uv sync --extra cpu

ENV LLM_STREAM_URL=http://ai-learning-assistant-training-server:3000/api/ai-chat/chat/stream
ENV APP_PORT=8989
ENV APP_HOST=0.0.0.0

EXPOSE 8989
WORKDIR /ai-learning-assistant-rtc-backend

CMD ["uv", "run", "main.py"]
