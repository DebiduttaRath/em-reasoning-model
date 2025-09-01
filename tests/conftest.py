# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from reasoning_from_scratch.qwen3 import (
    download_qwen3_small,
    QWEN_CONFIG_06_B,
    Qwen3Model,
)
import sys
import types
import nbformat
import pytest
import torch


@pytest.fixture(scope="session")
def qwen3_weights_path(tmp_path_factory):
    """Creates and saves a deterministic model for testing."""

    base_path = tmp_path_factory.mktemp("models")
    model_path = base_path / "qwen3_test_weights.pt"
    download_qwen3_small(kind="base", tokenizer_only=True, out_dir=base_path)

    if not model_path.exists():
        torch.manual_seed(123)
        model = Qwen3Model(QWEN_CONFIG_06_B)
        torch.save(model.state_dict(), model_path)

    return base_path


def import_definitions_from_notebook(nb_path, module_name):
    if not nb_path.exists():
        raise FileNotFoundError(f"Notebook file not found at: {nb_path}")

    nb = nbformat.read(str(nb_path), as_version=4)

    mod = types.ModuleType(module_name)
    sys.modules[module_name] = mod

    # Pass 1: execute only imports (handle multi-line)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        lines = cell.source.splitlines()
        collecting = False
        buf = []
        paren_balance = 0
        for line in lines:
            stripped = line.strip()
            if not collecting and (stripped.startswith("import ") or stripped.startswith("from ")):
                collecting = True
                buf = [line]
                paren_balance = line.count("(") - line.count(")")
                if paren_balance == 0:
                    exec("\n".join(buf), mod.__dict__)
                    collecting = False
                    buf = []
            elif collecting:
                buf.append(line)
                paren_balance += line.count("(") - line.count(")")
                if paren_balance == 0:
                    exec("\n".join(buf), mod.__dict__)
                    collecting = False
                    buf = []

    # Pass 2: execute only def/class definitions
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source
        if "def " in src or "class " in src:
            exec(src, mod.__dict__)

    return mod
