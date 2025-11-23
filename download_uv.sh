#!/usr/bin/bash
# Linux x86_64
mkdir -p uv_executable
pushd uv_executable
wget -O uv.tar.gz https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz
tar xf uv.tar.gz
rm uv.tar.gz
popd

# macOS x86_64  
mkdir -p uv_executable
pushd uv_executable
wget -O uv.tar.gz https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-apple-darwin.tar.gz
tar xf uv.tar.gz
rm uv.tar.gz
popd

# macOS ARM64
mkdir -p uv_executable
pushd uv_executable
wget -O uv.tar.gz https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz
tar xf uv.tar.gz
rm uv.tar.gz
popd

# Windows
mkdir -p uv_executable/uv-x86_64-pc-windows-msvc
pushd uv_executable/uv-x86_64-pc-windows-msvc
wget -O uv.zip https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip
unzip uv.zip
rm uv.zip
popd
