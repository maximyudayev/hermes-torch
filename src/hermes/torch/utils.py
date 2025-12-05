############
#
# Copyright (c) 2024-2025 Maxim Yudayev and KU Leuven eMedia Lab
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
import sys
from torch.nn import Module
from torch import load


def build_model(
    module_path: str,
    module_name: str,
    class_name: str,
    module_params: dict,
    checkpoint_path: str,
) -> Module:
    model_class: type[Module] = search_model_class(module_path, class_name, module_name)
    model: Module = model_class(**module_params)
    model.load_state_dict(load(checkpoint_path))
    return model


def search_model_class(
    module_path: str, class_name: str, module_name: str
) -> type[Module]:
    try:
        module_spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_name] = module
        module_spec.loader.exec_module(module)
    except ImportError as e:
        raise ImportError("Could not import subpackage '%s'" % (module)) from e

    if not hasattr(module, class_name):
        raise AttributeError(
            "Class '%s' not found in module '%s'. "
            "Check the spelling of the class name." % (class_name, module)
        )

    class_type = getattr(module, class_name)
    return class_type
