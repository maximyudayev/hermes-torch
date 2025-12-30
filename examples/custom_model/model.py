############
#
# Copyright (c) 2025 Vayalet Stefanova and KU Leuven eMedia Lab
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
# Created 2025 for AidWear, AidFOG, and RevalExo projects of KU Leuven.
#
# ############

from collections import deque
from typing import Any
import torch
from torch.nn import Module
import pytorch_tcn
import numpy as np

from utils import init_iir_filter, smooth, normalize


class TCN(Module):
    def __init__(
        self,
        input_size: tuple[int, int],
        output_classes: list[str],
        sampling_rate_hz: int,
    ):
        super().__init__()
        self._tcn: Module = pytorch_tcn.TCN(
            num_inputs=30,
            num_channels=[16, 32, 32, 32, 16],
            kernel_size=3,
            dropout=0.1,
            output_projection=2,
            output_activation=None,
            causal=True,
        )

        # Initialize highpass filter.
        self._b, self._a, self._zi = init_iir_filter(
            fs=sampling_rate_hz, cutoff_hz=0.3, order=4, num_channels=input_size[1]
        )

        # Initialize vars for pre-processing: (x-mean)/std.
        self._mean = np.zeros(6, dtype=np.float32)
        self._var = np.ones(6, dtype=np.float32)
        self._count = 0

        # State for label smoothing.
        self._smooth_state = (False, 0, 0)  # (in_fog, consec_ones, consec_zeros)

        # Buffer latest valid sample of each sensor.
        self._buffers: list[deque[Any]] = [
            deque([np.zeros(input_size[1])], maxlen=1) for _ in range(input_size[0])
        ]

    def forward(self, data):
        acc = data["dots-imu"]["acceleration"]
        gyr = data["dots-imu"]["gyroscope"]

        for i, sample in enumerate(np.concatenate((acc, gyr), axis=1)):
            if all(map(lambda el: not np.isnan(el), sample)):
                # Pre-process valid sample.
                norm_sample, self._zi, self._count, self._mean, self._var = normalize(
                    sample,
                    self._b,
                    self._a,
                    self._zi,
                    self._count,
                    self._mean,
                    self._var,
                )
                self._buffers[i].append(norm_sample)

        x = torch.tensor(
            np.concatenate([buf[-1] for buf in self._buffers]), dtype=torch.float32
        )[None, :, None]

        y: torch.Tensor = self._tcn(x)
        logits = y.squeeze().numpy()
        prediction = y.squeeze().argmax().item()
        prediction, self._smooth_state = smooth(prediction, self._smooth_state)

        return logits, prediction
