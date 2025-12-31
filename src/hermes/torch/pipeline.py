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

from multiprocessing import Process, Event, Queue
from queue import Empty
import numpy as np

from hermes.base.nodes.pipeline import Pipeline
from hermes.utils.types import LoggingSpec
from hermes.utils.mp_utils import launch_handler
from hermes.utils.zmq_utils import (
    PORT_BACKEND,
    PORT_FRONTEND,
    PORT_SYNC_HOST,
    PORT_KILL,
)

from .stream import TorchClassifierStream
from .handler import TorchClassifierHandler


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
        port_pub: str = PORT_BACKEND,
        port_sub: str = PORT_FRONTEND,
        port_sync: str = PORT_SYNC_HOST,
        port_killsig: str = PORT_KILL,
        **_,
    ):
        # Wrap AI compute in a separate process to not stall ZeroMQ.
        self._is_ready_event = Event()
        self._is_keep_data_event = Event()
        self._is_stop_new_data_event = Event()
        self._is_cleanup_event = Event()
        self._is_finished_event = Event()
        self._input_queue: Queue[tuple[str, dict]] = Queue()
        self._output_queue: Queue[tuple[np.ndarray, int, float, float]] = Queue()

        self._handler_proc = Process(
            target=launch_handler,
            args=(TorchClassifierHandler,),
            kwargs={
                "ref_time_s": logging_spec.ref_time_s,
                "file_path": stream_out_spec["file_path"],
                "class_name": stream_out_spec["class_name"],
                "module_params": stream_out_spec["module_params"],
                "checkpoint_path": stream_out_spec["checkpoint_path"],
                "is_ready_event": self._is_ready_event,
                "is_keep_data_event": self._is_keep_data_event,
                "is_stop_new_data_event": self._is_stop_new_data_event,
                "is_cleanup_event": self._is_cleanup_event,
                "is_finished_event": self._is_finished_event,
                "input_queue": self._input_queue,
                "output_queue": self._output_queue,
            },
        )
        self._handler_proc.start()
        self._is_ready_event.wait()

        stream_out_spec = {
            "classes": stream_out_spec["module_params"]["output_classes"],
            "sampling_rate_hz": stream_out_spec["module_params"]["sampling_rate_hz"],
        }

        super().__init__(
            host_ip=host_ip,
            stream_out_spec=stream_out_spec,
            stream_in_specs=stream_in_specs,
            logging_spec=logging_spec,
            is_async_generate=True,
            port_pub=port_pub,
            port_sub=port_sub,
            port_sync=port_sync,
            port_killsig=port_killsig,
        )

    @classmethod
    def create_stream(cls, stream_spec: dict) -> TorchClassifierStream:
        return TorchClassifierStream(**stream_spec)

    def _keep_samples(self):
        self._is_keep_data_event.set()

    def _process_data(self, topic: str, msg: dict) -> None:
        # TODO: put the data into the user-selected type of alignment mechanism, before pushing into PyTorch subprocess.
        self._input_queue.put_nowait((topic, msg["data"]))

    def _generate_data(self) -> None:
        try:
            # Get data from the AI model without blocking.
            logits, prediction, start_time_s, end_time_s = self._output_queue.get_nowait()
            data = {
                "prediction": prediction,
                "logits": logits,
                "compute_time_s": end_time_s - start_time_s,
            }
            tag: str = "%s.data" % self._log_source_tag()
            self._publish(tag, process_time_s=end_time_s, data={"classifier": data})
        except Empty:
            if self._is_finished_event.is_set():
                self._notify_no_more_data_out()

    def _stop_new_data(self) -> None:
        self._is_stop_new_data_event.set()

    def _cleanup(self) -> None:
        self._is_cleanup_event.set()
        self._handler_proc.join()
        super()._cleanup()
