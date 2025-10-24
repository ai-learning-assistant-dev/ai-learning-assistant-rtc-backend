# 中文实时AI对话后端

使用FastRTC框架搭建，由ASR、VAD、TTS和LLM组成

## 安装依赖

对于python的包，可以直接用`pip`管理或用`uv`管理

> [!NOTE]
> 推荐使用`uv`来管理依赖，它支持并发下载依赖，速度比`pip`快很多

### 用pip安装依赖

如果你没有虚拟环境，可以用如下方式创建一个虚拟环境：

```bash
python -m venv [虚拟环境名]
```

#### 安装PyTorch、FunASR和kokoro

如果是`python -m venv`创建的虚拟环境，则需要先激活虚拟环境：

```bash
# 如果是bash/zsh:
source [虚拟环境名]/bin/activate
# 如果是fish:
source [虚拟环境名]/bin/activate.fish
# 如果是powershell:
source [虚拟环境名]/bin/Activate.ps1
```

用`pip`安装`PyTorch`、`FunASR`和`kokoro`时需要根据当前设备图形加速器类型决定安装哪个版本：

```bash
# CPU版本：
pip3 install torch torchvision funasr kokoro --index-url https://download.pytorch.org/whl/cpu
# NVIDIA CUDA 12.8 版本（其他CUDA版本依次类推）：
pip3 install torch torchvision funasr kokoro --index-url https://download.pytorch.org/whl/cu128
# AMD ROCm 6.4 版本：
pip3 install torch torchvision funasr kokoro --index-url https://download.pytorch.org/whl/rocm6.4
```

#### 其他依赖

在创建虚拟环境后可以启用虚拟环境、用`requirements.txt`自动下载依赖

先激活虚拟环境：

```bash
# 如果是bash/zsh:
source [虚拟环境名]/bin/activate
# 如果是fish:
source [虚拟环境名]/bin/activate.fish
# 如果是powershell:
source [虚拟环境名]/bin/Activate.ps1
```

此后命令行会有虚拟环境的标识符，观察到后执行：

```bash
pip install -r requirements.txt
```

### 用uv安装依赖

如果使用CPU版本的`PyTorch`，可以直接执行如下命令：

```bash
uv sync
```

如果想更换`PyTorch`版本，`uv`提供了自动识别当前系统环境来决定安装的`PyTorch`版本的能力：

```bash
uv pip install torch torchaudio funasr kokoro --torch-backend=auto
```

> [!NOTE]
> 如果你的电脑有NVIDIA显卡但没有安装CUDA，它也默认会安装CPU版本

### 环境变量

目前的`main.py`需要`DEEPSEEK_API_KEY`（如果你能联系到我，我可以单独发给你我的API key），设置好才能运行LLM

## 运行后端代码

启动虚拟环境后，运行：

```bash
python main.py
```

如果是`uv`也可以这样运行：

```bash
uv run main.py
```

## 杂项

### 离线化运行（不依赖远程模型下载）

目前已经做到TTS和STT(ASR)离线化了，但是VAD暂时用的是FastRTC的内置实现，还需要联网（且需要可稳定访问PyPi的环境）

#### 下载kokoro的权重文件

> [!IMPORTANT]
> 仓库已经打包好了这些权重文件，您不需要执行这些操作

原本的代码实现了自动下载模型，但是每一次使用都会调用模型下载，导致国内网络环境下无法执行，故采用本地权重文件的形式保存

- [模型权重kokoro-v1_1-zh.pth下载地址](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/kokoro-v1_1-zh.pth?download=true)
- [中文声音权重zf_001.pt下载地址](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/voices/zf_001.pt?download=true)
- [配置文件config.json下载地址](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/resolve/main/config.json?download=true)

将它们放到`kokoro_tts`文件夹中，即可运行

> [!NOTE]
> 虽然权重这种大型二进制文件不建议用git管理，但是自动化管理总比手动下载方便
