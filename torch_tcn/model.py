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
import pytorch_tcn
import numpy as np

from .ai_utils import init_iir_filter, smooth, normalize

class TCN(Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        output_classes: list[str],
        sampling_rate_hz: int,
    ):

        self._model: Module = pytorch_tcn.TCN(
            num_inputs=30,
            num_channels=[16, 32, 32, 32, 16],
            kernel_size=3,
            dropout=0.1,
            output_projection=2,
            output_activation=None,
            causal=True,
        )

        # Initialize highpass filter
        self.b, self.a, self.zi = init_iir_filter(
            fs=sampling_rate_hz, cutoff_hz=0.3, order=4, num_channels=input_size[1]
        )

        # Initialize vars needed for pre-processing: (x-mean)/std
        self._mean = np.zeros(6, dtype=np.float32)
        self._var = np.ones(6, dtype=np.float32)
        self._count = 0

        # to keep state for label smoothing
        self.smooth_state = (False, 0, 0)  # (in_fog, consec_ones, consec_zeros)

    def _generate_prediction(self) -> tuple[list[float], int]:
        input_tensor = torch.tensor(
            np.concatenate([buf[0] for buf in self._buffer]), dtype=torch.float32
        )[None, :, None]
        output = self._model(input_tensor, inference=True)
        logits = output.squeeze().numpy()
        prediction = output.squeeze().argmax().item()
        prediction, self.smooth_state = smooth(prediction, self.smooth_state)
        return logits, prediction

    def _process_data(self, topic: str, msg: dict) -> None:
        acc = msg["data"]["dots-imu"]["acceleration"]
        gyr = msg["data"]["dots-imu"]["gyroscope"]
        toa_s = msg["data"]["dots-imu"]["toa_s"]

        for i, sensor_sample in enumerate(np.concatenate((acc, gyr), axis=1)):
            if all(map(lambda el: not np.isnan(el), sensor_sample)):
                # pre-process valid sample
                norm_sample, self.zi, self._count, self._mean, self._var = normalize(
                    sensor_sample,
                    self.b,
                    self.a,
                    self.zi,
                    self._count,
                    self._mean,
                    self._var,
                )
                self._buffer[i].append(norm_sample)
                # self._buffer[i].append(sensor_sample)
