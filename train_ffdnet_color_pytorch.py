# -*- coding: utf-8 -*-
"""彩色 FFDNet 训练入口。

该文件只设置 color 训练默认参数，实际 FFDNet 网络、加噪和优化逻辑仍在 train_ffdnet_pytorch.py 中。
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


if "--channels" in sys.argv:
    idx = sys.argv.index("--channels")
    value = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None
    if value != "3":
        raise ValueError("train_ffdnet_color_pytorch.py 是彩色训练入口，--channels 必须为 3。")
else:
    sys.argv.extend(["--channels", "3"])
if "--save_dir" not in sys.argv:
    sys.argv.extend(["--save_dir", "models/FFDNet_DFWB_color"])

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
runpy.run_path(str(script_dir / "train_ffdnet_pytorch.py"), run_name="__main__")
