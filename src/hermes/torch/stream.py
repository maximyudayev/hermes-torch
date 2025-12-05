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

from collections import OrderedDict

from hermes.base.stream import Stream


class TorchStream(Stream):
    """A structure to store PyTorch prediction outputs.

    TODO: use user parameters to specify model
      output configuration
      (i.e. classifier, regressor, embedding, etc.)
    """

    def __init__(self, classes: list[str], sampling_rate_hz: float, **_) -> None:
        super().__init__()

        self._classes = classes
        self._device_name = "pytorch"
        self._define_data_notes()

        self.add_stream(
            device_name=self._device_name,
            stream_name="prediction",
            data_type="uint16",
            sample_size=(1,),
            sampling_rate_hz=sampling_rate_hz,
            data_notes=self._data_notes[self._device_name]["prediction"],
            is_measure_rate_hz=True,
        )
        self.add_stream(
            device_name=self._device_name,
            stream_name="logits",
            data_type="float64",
            sample_size=(len(classes),),
            data_notes=self._data_notes[self._device_name]["logits"],
        )

    def get_fps(self) -> dict[str, float | None]:
        return {self._device_name: super()._get_fps(self._device_name, "prediction")}

    def _define_data_notes(self) -> None:
        self._data_notes = {}
        self._data_notes.setdefault(self._device_name, {})

        self._data_notes[self._device_name]["logits"] = OrderedDict(
            [
                ("Description", "Probability vector"),
                ("Range", "[0,1]"),
                (Stream.metadata_data_headings_key, self._classes),
            ]
        )
        self._data_notes[self._device_name]["prediction"] = OrderedDict(
            [
                ("Description", "Label of the most likely class prediction"),
            ]
        )
