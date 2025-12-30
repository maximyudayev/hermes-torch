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

import torch
from torch.nn import Module

from hermes.base.nodes.pipeline import Pipeline
from hermes.utils.types import LoggingSpec
from hermes.utils.time_utils import get_time
from hermes.utils.zmq_utils import (
    PORT_BACKEND,
    PORT_FRONTEND,
    PORT_SYNC_HOST,
    PORT_KILL,
)

from .stream import TorchClassifierStream
from .utils import build_model


class TorchClassifierPipeline(Pipeline):
    """A class for processing realtime sensor data with a custom PyTorch AI model."""

    # TODO: make the log name assignable.
    @classmethod
    def _log_source_tag(cls) -> str:
        return "ai"

    def __init__(
        self,
        host_ip: str,
        stream_out_spec: dict,
        stream_in_specs: list[dict],
        logging_spec: LoggingSpec,
        is_async_generate: bool = False,
        port_pub: str = PORT_BACKEND,
        port_sub: str = PORT_FRONTEND,
        port_sync: str = PORT_SYNC_HOST,
        port_killsig: str = PORT_KILL,
        **_,
    ):
        # TODO: wrap AI compute in a separate process to not stall ZeroMQ.
        self._model: Module = build_model(
            file_path=stream_out_spec["file_path"],
            class_name=stream_out_spec["class_name"],
            module_params=stream_out_spec["module_params"],
            checkpoint_path=stream_out_spec["checkpoint_path"],
        )
        # Inference-only mode.
        self._model.eval()
        # Globally turn off gradient accumulation.
        torch.set_grad_enabled(False)

        stream_out_spec = {
            "classes": stream_out_spec["module_params"]["output_classes"],
            "sampling_rate_hz": stream_out_spec["module_params"]["sampling_rate_hz"],
        }

        super().__init__(
            host_ip=host_ip,
            stream_out_spec=stream_out_spec,
            stream_in_specs=stream_in_specs,
            logging_spec=logging_spec,
            is_async_generate=is_async_generate,
            port_pub=port_pub,
            port_sub=port_sub,
            port_sync=port_sync,
            port_killsig=port_killsig,
        )

    @classmethod
    def create_stream(cls, stream_spec: dict) -> TorchClassifierStream:
        return TorchClassifierStream(**stream_spec)

    def _keep_samples(self):
        pass

    def _process_data(self, topic: str, msg: dict) -> None:
        # TODO: put data into the IPC queues for PyTorch subprocess to asynchronously compute and not stall ZeroMQ.

        start_time_s: float = get_time()
        logits, prediction = self._model(msg["data"])
        end_time_s: float = get_time()

        data = {
            "prediction": prediction,
            "logits": logits,
            "compute_time_s": end_time_s - start_time_s,
        }
        tag: str = "%s.data" % self._log_source_tag()

        self._publish(tag, process_time_s=end_time_s, data={"classifier": data})

    def _generate_data(self):
        pass

    def _stop_new_data(self):
        pass

    def _cleanup(self) -> None:
        super()._cleanup()
