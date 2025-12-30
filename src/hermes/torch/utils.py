############
#
# Copyright (c) 2024-2026 Maxim Yudayev and KU Leuven eMedia Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Created 2024-2025 for the KU Leuven AidWear, AidFOG, and RevalExo projects
# by Maxim Yudayev [https://yudayev.com].
#
# ############

import importlib
import importlib.util
import os
import sys
from torch.nn import Module
from torch import load


def build_model(
    file_path: str,
    class_name: str,
    module_params: dict,
    checkpoint_path: str,
) -> Module:
    model_class: type[Module] = search_model_class(file_path, class_name)
    model: Module = model_class(**module_params)
    model.load_state_dict(load(checkpoint_path))
    return model


def search_model_class(file_path: str, class_name: str) -> type[Module]:
    file_path = os.path.abspath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.endswith(".py"):
        raise ValueError(f"File must be a .py file: {file_path}")

    parent_dir = os.path.dirname(file_path)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

    module_name = f"hermes.torch.custom_model.{class_name}"

    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module spec from {file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise ImportError(f"Failed to import module from {file_path}: {str(e)}") from e

    if not hasattr(module, class_name):
        available_classes = [name for name in dir(module) if not name.startswith("_")]
        raise AttributeError(
            f"Class '{class_name}' not found in {file_path}. "
            f"Available names: {', '.join(available_classes)}"
        )

    class_type = getattr(module, class_name)

    if not isinstance(class_type, type):
        raise TypeError(
            f"'{class_name}' is not a class, it's a {type(class_type).__name__}"
        )

    return class_type
