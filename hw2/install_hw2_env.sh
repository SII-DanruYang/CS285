#!/usr/bin/env bash
set -euo pipefail

ASSUME_YES="${CS285_INSTALL_ASSUME_YES:-0}"

confirm() {
  local prompt="$1"

  if [[ "$ASSUME_YES" == "1" ]]; then
    return 0
  fi

  read -r -p "$prompt [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

cat <<'WARNING'
警告：此脚本将执行以下操作：
1. 安装 Linux 系统软件包；
2. 创建 Conda 环境 cs285_hw2；
3. 如果同名环境已经存在，将先删除该环境；
4. 安装课程指定的旧版 Python 依赖。

请勿在包含重要未备份环境的机器上直接运行。
设置 CS285_INSTALL_ASSUME_YES=1 可以跳过交互确认。
WARNING

confirm "确认继续执行吗？" || {
  echo "已取消安装。"
  exit 0
}


echo "========== 0. 检查是否在 hw2 目录 =========="
if [ ! -f "requirements.txt" ] || [ ! -d "cs285" ]; then
  echo "错误：请先 cd 到 homework_fall2023/hw2 目录，再运行本脚本。"
  exit 1
fi

echo "========== 1. 安装 Linux 系统依赖 =========="
if command -v apt-get >/dev/null 2>&1; then
  if [[ "$EUID" -eq 0 ]]; then
    APT_COMMAND=(apt-get)
  elif command -v sudo >/dev/null 2>&1; then
    APT_COMMAND=(sudo apt-get)
  else
    echo "错误：安装系统依赖需要 root 或 sudo 权限。"
    exit 1
  fi

  "${APT_COMMAND[@]}" update
  DEBIAN_FRONTEND=noninteractive "${APT_COMMAND[@]}" install -y \
    swig \
    ffmpeg \
    xvfb \
    patchelf \
    libgl1 \
    libegl1 \
    libgles2 \
    libosmesa6-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libfreetype6-dev \
    libportmidi-dev
else
  echo "没有 apt-get，跳过系统依赖安装。"
fi

echo "========== 2. 初始化 conda =========="
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "========== 3. 检查旧的 cs285_hw2 环境 =========="
if conda env list | awk '{print $1}' | grep -qx "cs285_hw2"; then
  echo "发现已有的 cs285_hw2 环境，将按用户确认结果删除。"
  conda env remove -n cs285_hw2 -y
else
  echo "未发现已有的 cs285_hw2 环境。"
fi

echo "========== 4. 创建 Python 3.10 环境 =========="
conda create -n cs285_hw2 python=3.10 -y
conda activate cs285_hw2

echo "========== 5. 升级 pip 基础工具 =========="
python -m pip install --upgrade pip setuptools wheel

echo "========== 6. 先用 conda 安装容易编译失败的包 =========="
conda install -c conda-forge -y \
  swig=4.0.2 \
  pygame \
  box2d-py \
  ffmpeg \
  pyopengl \
  glfw \
  glew \
  patchelf

echo "========== 7. 安装 HW2 Python 依赖 =========="
# 这里不用 gym[box2d,classic_control]，而是 gym + conda 的 pygame/box2d-py
# 这样可以绕开你刚才 pygame 编译 SDL2 报错的问题。
python -m pip install \
  mujoco==2.2.0 \
  gym==0.25.2 \
  tensorboard==2.10.0 \
  tensorboardX==2.5.1 \
  matplotlib==3.5.3 \
  ipython==7.34.0 \
  moviepy==1.0.3 \
  pyvirtualdisplay==3.0 \
  torch==1.13.1 \
  opencv-python==4.6.0.66 \
  ipdb==0.13.9

echo "========== 8. 安装当前 hw2 项目本身 =========="
python -m pip install -e .

echo "========== 9. 设置无显示器环境下 MuJoCo 渲染 =========="
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh" <<'EOF'
export MUJOCO_GL=egl
EOF
export MUJOCO_GL=egl

echo "========== 10. 测试核心库 =========="
python - <<'PY'
import sys
print("python:", sys.version)

import numpy
print("numpy ok:", numpy.__version__)

import torch
print("torch ok:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

import gym
print("gym ok:", gym.__version__)

import mujoco
print("mujoco ok:", mujoco.__version__)

import pygame
print("pygame ok:", pygame.version.ver)

import Box2D
print("Box2D ok")

import cv2
print("opencv ok:", cv2.__version__)

import tensorboardX
print("tensorboardX ok")

print("========== 测试环境 ==========")

env = gym.make("CartPole-v0")
print("CartPole-v0 ok")
env.close()

env = gym.make("LunarLander-v2")
print("LunarLander-v2 ok")
env.close()

env = gym.make("HalfCheetah-v4")
print("HalfCheetah-v4 ok")
env.close()

print("全部依赖安装成功！")
PY

echo ""
echo "安装完成。以后每次跑 HW2 前执行："
echo "conda activate cs285_hw2"
