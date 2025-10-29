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

#### 安装各种PyTorch（适配不同显卡）

可以根据设备不同选择执行如下命令：

```bash
uv sync --extra cpu     # CPU 用户
uv sync --extra cu128   # NVIDIA GPU 用户
uv sync --extra rocm64  # AMD GPU 用户
```

> [!NOTE]
> 虽然我们支持安装AMD显卡版本的PyTorch，但你需要手动在系统安装`rocrand`才能运行本项目。
>
> 另外，ROCm对APU的iGPU适配比较欠缺，由于iGPU显存和系统内存共用而频繁触发页迁移，在780M上效果非常差，推理速度可能不如CPU。

如果已经安装过某个版本想用新版本覆盖，直接执行：

```bash
uv sync --extra [目标版本]    # 这里目标版本只有cpu, cu128, rocm64三种选项
```

这样就可以自动覆盖原本的安装依赖了

> [!NOTE]
> 如果覆盖后运行出现找不到`kokoro`或`misaki[zh]`包的报错，可以尝试：
>
> ```bash
> rm -r .venv
> uv sync --extra [目标版本]
> ```

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
