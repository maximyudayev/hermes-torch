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

import numpy as np
import torch
from torch.nn import Module
from multiprocessing import Queue
from multiprocessing.synchronize import Event as _Event
from queue import Empty

from hermes.utils.time_utils import get_time, init_time

from .utils import build_model


class TorchClassifierHandler:
    def __init__(
        self,
        ref_time_s: float,
        file_path: str,
        class_name: str,
        module_params: dict,
        checkpoint_path: str,
        is_ready_event: _Event,
        is_keep_data_event: _Event,
        is_stop_new_data_event: _Event,
        is_cleanup_event: _Event,
        is_finished_event: _Event,
        input_queue: "Queue[tuple[str, dict]]",
        output_queue: "Queue[tuple[np.ndarray, int, float, float]]",
    ):
        self._ref_time_s = ref_time_s
        self._is_ready_event = is_ready_event
        self._is_keep_data_event = is_keep_data_event
        self._is_stop_new_data_event = is_stop_new_data_event
        self._is_cleanup_event = is_cleanup_event
        self._is_finished_event = is_finished_event
        self._input_queue = input_queue
        self._output_queue = output_queue

        self._model: Module = build_model(
            file_path=file_path,
            class_name=class_name,
            module_params=module_params,
            checkpoint_path=checkpoint_path,
        )
        # Inference-only mode.
        self._model.eval()
        # Globally turn off gradient accumulation.
        torch.set_grad_enabled(False)
        # TODO: compile the model for higher efficiency.

    def __call__(self) -> None:
        init_time(ref_time=self._ref_time_s)
        self._is_ready_event.set()
        self._is_keep_data_event.wait()

        while True:
            try:
                topic, data = self._input_queue.get(timeout=10)
                start_time_s: float = get_time()
                result = self._model(topic, data)
                end_time_s: float = get_time()
                if result is not None:
                    (logits, prediction) = result
                    self._output_queue.put((logits, prediction, start_time_s, end_time_s))
            except Empty:
                print("PyTorch input queue timed out, continues checking for stopping.", flush=True)
                if self._is_stop_new_data_event.is_set():
                    self._is_finished_event.set()
                    break

        self._is_cleanup_event.wait()
        print("PyTorch subprocess exited.", flush=True)
