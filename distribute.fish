#!/usr/bin/fish
function distribute_unix -a target_triple
    # local variable
    set -l target_dir rtc_backend-$target_triple
    mkdir $target_dir
    pushd $target_dir
    git clone https://github.com/ai-learning-assistant-dev/ai-learning-assistant-rtc-backend.git .
    rm -rf .git
    mv pyproject.dist.toml pyproject.toml
    cp ../en_core_web_sm-3.8.0-py3-none-any.whl .
    wget -O uv.tar.gz https://github.com/astral-sh/uv/releases/latest/download/uv-$target_triple.tar.gz
    tar xf uv.tar.gz
    rm uv.tar.gz
    mv uv-$target_triple uv_executable
    popd
    tar czf rtc_backend-$target_triple.tar.gz $target_dir
    rm -rf $target_dir
end

function distribute_windows -a target_triple
    mkdir rtc_backend-$target_triple
    pushd rtc_backend-$target_triple
    git clone https://github.com/ai-learning-assistant-dev/ai-learning-assistant-rtc-backend.git .
    rm -rf .git
    mv pyproject.dist.toml pyproject.toml
    cp ../en_core_web_sm-3.8.0-py3-none-any.whl .
    wget -O uv.zip https://github.com/astral-sh/uv/releases/latest/download/uv-$target_triple.zip
    unzip uv.zip
    rm uv.zip
    popd
    tar czf rtc_backend-$target_triple.zip rtc_backend-$target_triple
    rm -rf rtc_backend-$target_triple
end

wget https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# x86_64 Linux
distribute_unix x86_64-unknown-linux-gnu

# x86_64 MacOS
distribute_unix x86_64-apple-darwin

# Arm64 MacOS
distribute_unix aarch64-apple-darwin

# x86_64 windows
distribute_windows x86_64-pc-windows-msvc

rm en_core_web_sm-3.8.0-py3-none-any.whl
