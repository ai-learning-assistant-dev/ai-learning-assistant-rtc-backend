#!/usr/bin/bash
wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# x86_64 Linux
mkdir rtc_backend-x86_64-unknown-linux-gnu
pushd rtc_backend-x86_64-unknown-linux-gnu
git clone https://github.com/ai-learning-assistant-dev/ai-learning-assistant-rtc-backend.git .
cp ../en_core_web_sm-3.8.0-py3-none-any.whl .
wget -O uv.tar.gz https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
tar xf uv.tar.gz
rm uv.tar.gz
mv uv-x86_64-unknown-linux-gnu uv_executable
popd
tar czf rtc_backend-x86_64-unknown-linux-gnu.tar.gz rtc_backend-x86_64-unknown-linux-gnu
rm -rf rtc_backend-x86_64-unknown-linux-gnu

# x86_64 MacOS
mkdir rtc_backend-x86_64-apple-darwin
pushd rtc_backend-x86_64-apple-darwin
git clone https://github.com/ai-learning-assistant-dev/ai-learning-assistant-rtc-backend.git .
cp ../en_core_web_sm-3.8.0-py3-none-any.whl .
wget -O uv.tar.gz https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-apple-darwin.tar.gz
tar xf uv.tar.gz
rm uv.tar.gz
mv uv-x86_64-apple-darwin uv_executable
popd
tar czf rtc_backend-x86_64-apple-darwin.tar.gz rtc_backend-x86_64-apple-darwin
rm -rf rtc_backend-x86_64-apple-darwin 

# Arm64 MacOS
mkdir rtc_backend-aarch64-apple-darwin
pushd rtc_backend-aarch64-apple-darwin
git clone https://github.com/ai-learning-assistant-dev/ai-learning-assistant-rtc-backend.git .
cp ../en_core_web_sm-3.8.0-py3-none-any.whl .
wget -O uv.tar.gz https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz
tar xf uv.tar.gz
rm uv.tar.gz
mv uv-aarch64-apple-darwin uv_executable
popd
tar czf rtc_backend-aarch64-apple-darwin.tar.gz rtc_backend-aarch64-apple-darwin 
rm -rf rtc_backend-aarch64-apple-darwin 

# x86_64 windows
mkdir rtc_backend-x86_64-pc-windows-msvc
pushd rtc_backend-x86_64-pc-windows-msvc
git clone https://github.com/ai-learning-assistant-dev/ai-learning-assistant-rtc-backend.git .
cp ../en_core_web_sm-3.8.0-py3-none-any.whl .
wget -O uv.zip https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip
unzip uv.zip
rm uv.zip
popd
tar czf rtc_backend-x86_64-pc-windows-msvc.tar.gz rtc_backend-x86_64-pc-windows-msvc 
rm -rf rtc_backend-x86_64-pc-windows-msvc 

rm en_core_web_sm-3.8.0-py3-none-any.whl
