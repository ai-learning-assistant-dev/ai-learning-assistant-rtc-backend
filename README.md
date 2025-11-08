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

#### 根据硬件情况安装不同依赖

如果是`python -m venv`创建的虚拟环境，则需要先激活虚拟环境：

```bash
# 如果是bash/zsh:
source [虚拟环境名]/bin/activate
# 如果是fish:
source [虚拟环境名]/bin/activate.fish
# 如果是powershell:
source [虚拟环境名]/bin/Activate.ps1
```

```bash
# CPU版本：
pip install -r requirements-cpu.txt
# NVIDIA CUDA 12.8 版本：
pip install -r requirements-cuda.txt
# AMD ROCm 6.4 版本：
pip install -r requirements-rocm.txt
```

#### 修改CUDA/ROCm版本

打开相应的`requirements-[cuda|rocm].txt`，修改下面这一行：

```txt
--extra-index-url https://download.pytorch.org/whl/[source]
```

例如CUDA13.0就修改为：

```txt
--extra-index-url https://download.pytorch.org/whl/cu130
```

### 用uv安装依赖

#### 根据硬件情况安装依赖

可以根据设备不同选择执行如下命令：

```bash
uv sync --extra cpu     # CPU 用户
uv sync --extra cu128   # NVIDIA GPU 用户
uv sync --extra rocm64  # AMD GPU 用户
```

> [!IMPORTANT]
> AMD的ROCm目前**只适配了Linux版本**的，其他系统暂时无法运行。
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

## 运行代码

启动虚拟环境后，运行：

```bash
python main.py
```

如果是`uv`也可以这样运行：

```bash
uv run main.py
```

## 容器运行

Docker 运行方式：

```bash
docker run -d -p 8989:8989 \
    -e LLM_STREAM_URL=http://[AI学习助手启动器后端URL]:[AI学习助手启动器后端端口]/api/ai-chat/chat/stream \
    yaqia/rtc-backend
```

> [!NOTE]
> 如果是默认docker网桥推荐使用172.17.0.1作为AI学习助手启动器后端URL，端口默认是3000

### 端口配置

- 默认容器内端口：8989
- 可以通过`APP_PORT`修改
- 运行时映射：`-p 8080:8989`或`-p 8080:$APP_PORT`
