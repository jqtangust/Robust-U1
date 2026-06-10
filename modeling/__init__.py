# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Robust-U1 / BAGEL 推理与建模代码（自 Stage2 ``flow_grpo/bagel`` 迁入，包名为 ``modeling``）。"""

from .inferencer import InterleaveInferencer

__all__ = ["InterleaveInferencer"]
