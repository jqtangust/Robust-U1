"""仓库根路径与 ``sys.path`` 引导。"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def ensure_repo_on_path() -> Path:
    """将 Robust-U1 根目录加入 ``sys.path``，便于 ``import modeling``。"""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT
