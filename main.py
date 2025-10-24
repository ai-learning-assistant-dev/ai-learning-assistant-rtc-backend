import os

from fastrtc import ReplyOnPause, Stream
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from kokoro_tts.tts_adapter import get_kokoro_v11_zh_model
from funasr_stt.stt_adapter import LocalFunASR

deepseek_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/v1"
)

# ASR - FunASR or whisper.cpp
stt_model = LocalFunASR()
tts_model = get_kokoro_v11_zh_model()

conversations: list[ChatCompletionMessageParam] = [
    {
        "role": "system",
        "content": "You are a speaking assistant. Please always respond in natural and concise Chinese. Do not use Markdown, do not output lists or symbols, only provide content suitable for reading aloud.",
    },
]

def realtime_conversation(audio):
    prompt = stt_model.stt(audio).strip()
    if not prompt or len(prompt) < 2:
        return

    print("PROMPT:", prompt)

    conversations.append({"role": "user", "content": prompt})

    response = deepseek_client.chat.completions.create(
        model="deepseek-chat",
        messages=conversations,
        max_tokens=200,
        stream=True,
    )
    result = str()
    buffer = ""
    for event in response:
        delta = event.choices[0].delta.content or ""
        buffer += delta
        result += delta
        if buffer.endswith(("。", ".", "!", "?", " ")):  # 碰到句号或停顿
            tts_model.stream_tts_sync(buffer)
            for chunk in tts_model.stream_tts_sync(buffer):
                yield chunk
            buffer = ""  # 清空继续积累
    if buffer:
        for audio_chunk in tts_model.stream_tts_sync(buffer):
            yield audio_chunk

    print("RESPONSE:", result)
    conversations.append({"role": "assistant", "content": result})


stream = Stream(ReplyOnPause(realtime_conversation), modality="audio", mode="send-receive")

_ = stream.ui.launch()
