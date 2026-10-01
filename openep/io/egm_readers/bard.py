"""Python port of the MATLAB BardFile loader.

This class is intentionally standalone — it does not depend on the existing
EGM/EGMLoader/EGMParser model and is not wired into any UI/controller.

It mirrors the MATLAB sources in ``matlab_code_to_replicate/``:

* ``BardFile.m``            class definition, constants, constructor
* ``loadBardFile.m``        file-loading pipeline
* ``getbarddetails.m``      header / channel-info parser
* ``createEmptyMemMap.m``   int16 storage allocation (kept in-memory here)
* ``egm.m``                 raw a2d -> physical-unit (mV) scaling
* ``chNames2Indices.m``     label-to-index lookup

The A/D data is stored as an int16 NumPy array shaped
``(NSamples, NChannels)`` to match the MATLAB ``memmapfile`` layout
(``'Format', {'int16' [NSamples NChannels] 'a2d'}``). Physical-unit
electrograms in mV are produced on demand by :meth:`BardFile.egm` using the
same scaling formula as MATLAB::

    mv = a2d * 2 / (2 ** ADCBITS) * ChRange[channel]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

class Bard:
    """Loads a Bard-exported text file (Python port of MATLAB ``BardFile``)."""

    # MATLAB: properties (SetAccess = 'protected')
    ADCBITS: int = 16
    STIMTHRESHOLD: int = 1000  # 5 V/ms

    def __init__(
        self,
        file_path: Union[str, Path],
        ch_pace: Optional[int] = None,
    ):
        self.file_name: Path = Path(file_path)
        if not self.file_name.is_file():
            raise FileNotFoundError(f"BARDFILE: file not found: {file_path}")

        self.start_time: Optional[str] = None
        self.end_time: Optional[str] = None
        self.n_channels: int = 0
        self.n_samples: int = 0
        self.sample_rate: int = 0
        self.ch_name: List[str] = []
        self.ch_range: Optional[np.ndarray] = None
        self.ch_low: Optional[np.ndarray] = None
        self.ch_high: Optional[np.ndarray] = None
        self.ch_data_file_map: Optional[np.ndarray] = None

        # Parity with MATLAB Private* fields — populated but unused by load.
        self._ch_paced_in_file: str = "no entry in file"
        self._ch_stim: Optional[int] = None
        self._stim_indices: Optional[np.ndarray] = None
        self._is_stim_captured: Optional[np.ndarray] = None
        self._is_stim_indices_calculated: bool = False
        self._filter: List[int] = [100, 500]

        self.load_bard_file(ch_pace=ch_pace)

    @property
    def nyquist_freq(self) -> float:
        return float(self.sample_rate) / 2.0

    @property
    def short_file_name(self) -> str:
        return self.file_name.name

    @property
    def ch_stim(self) -> Optional[int]:
        return self._ch_stim

    @property
    def ch_stim_name(self) -> str:
        if self._ch_stim is None or self._ch_stim < 0:
            return ""
        return self.ch_name[self._ch_stim]

    @property
    def info(self) -> str:
        return (
            f"Type\t\t: BardFile\n"
            f"File Name\t\t: {self.short_file_name}\n"
            f"Start Time\t\t: {self.start_time}\n"
            f"End Time\t\t: {self.end_time}\n"
            f"N Channels\t\t: {self.n_channels}\n"
            f"N Samples\t\t: {self.n_samples}\n"
            f"Sample Rate Hz\t: {self.sample_rate}\n"
            f"Path\t\t: {self.file_name}\n"
            f"Ch Stim\t\t: {self.ch_stim_name or 'none'}\n"
        )

    def load_bard_file(self, ch_pace: Optional[int] = None) -> None:
        """Port of ``loadBardFile.m``."""
        if self.ch_data_file_map is not None:
            raise RuntimeError(
                "BARDFILE/LOADBARDFILE: there is already a datafilemap"
            )

        info, channels, data_start_byte = self._get_bard_details()

        self._ch_paced_in_file = info["ch_paced"]
        self.start_time = info["t_start"]
        self.end_time = info["t_end"]
        self.n_channels = info["n_channels"]
        self.n_samples = info["n_samples"]
        self.sample_rate = info["sample_rate"]

        if len(channels) != self.n_channels:
            raise ValueError(
                "BARDFILE/LOADBARDFILE: number of channel blocks "
                f"({len(channels)}) does not match Channels exported "
                f"({self.n_channels})"
            )

        self.ch_name = [""] * self.n_channels
        self.ch_range = np.zeros(self.n_channels, dtype=float)
        self.ch_low = np.zeros(self.n_channels, dtype=float)
        self.ch_high = np.zeros(self.n_channels, dtype=float)
        for i, ch in enumerate(channels):
            self.ch_name[i] = ch["label"]
            self.ch_range[i] = ch["range"]
            self.ch_low[i] = ch["low"]
            self.ch_high[i] = ch["high"]
            if ch["sample_rate"] != self.sample_rate:
                raise ValueError(
                    "BARDFILE/BARDFILE: Not all channels have the same sample rate!"
                )

        self.ch_data_file_map = self._read_data_block(data_start_byte)

        # MATLAB has a listdlg() branch for picking a paced channel when there
        # is no 'Channel paced:' header. We replicate only the CLI behaviour.
        if self._ch_paced_in_file == "no entry in file":
            self._ch_stim = ch_pace
        else:
            try:
                self._ch_stim = self.ch_name.index(self._ch_paced_in_file)
            except ValueError:
                self._ch_stim = None

    def _get_bard_details(self) -> Tuple[dict, List[dict], int]:
        """Port of ``getbarddetails.m``.

        Returns
        -------
        info
            Dict with keys ``sample_rate``, ``n_channels``, ``n_samples``,
            ``t_start``, ``t_end``, ``ch_paced``.
        channels
            List of per-channel dicts (``label``, ``range``, ``low``, ``high``,
            ``sample_rate``) in declaration order.
        data_start_byte
            Byte offset to the first data character, immediately after the
            ``[Data]`` line terminator. Matches MATLAB's ``ftell(fid)`` after
            ``[Data]``.
        """
        info = {
            "sample_rate": 0,
            "n_channels": 0,
            "n_samples": 0,
            "t_start": None,
            "t_end": None,
            "ch_paced": "no entry in file",
        }
        channels: List[dict] = []

        raw = self.file_name.read_bytes()
        data_match = re.search(rb"^\[Data\]\r?\n", raw, re.MULTILINE)
        if data_match is None:
            raise ValueError("BARDFILE/GETBARDDETAILS: [Data] marker not found")
        data_start_byte = data_match.end()
        header_text = raw[: data_match.start()].decode("ascii", errors="ignore")
        lines = header_text.splitlines()

        try:
            hdr_idx = next(i for i, ln in enumerate(lines) if ln.strip() == "[Header]")
        except StopIteration:
            raise ValueError("BARDFILE/GETBARDDETAILS: [Header] marker not found")

        current_channel: Optional[dict] = None
        for line in lines[hdr_idx + 1:]:
            stripped = line.strip()
            if not stripped:
                continue
            key_lower = stripped.lower()

            if current_channel is None:
                if key_lower.startswith("channel paced:"):
                    info["ch_paced"] = stripped[len("Channel paced:"):].strip()
                elif key_lower.startswith("channels exported:"):
                    info["n_channels"] = int(
                        stripped[len("Channels exported:"):].strip()
                    )
                elif key_lower.startswith("samples per channel:"):
                    info["n_samples"] = int(
                        stripped[len("Samples per channel:"):].strip()
                    )
                elif key_lower.startswith("start time:"):
                    info["t_start"] = stripped[len("Start time:"):].strip()
                elif key_lower.startswith("end time:"):
                    info["t_end"] = stripped[len("End time:"):].strip()
                elif key_lower.startswith("sample rate:"):
                    info["sample_rate"] = _parse_freq_hz(
                        stripped[len("Sample Rate:"):].strip()
                    )
                elif key_lower.startswith("channel #:"):
                    current_channel = _new_channel()
                    channels.append(current_channel)
                continue

            if key_lower.startswith("channel #:"):
                current_channel = _new_channel()
                channels.append(current_channel)
            elif key_lower.startswith("label:"):
                current_channel["label"] = stripped[len("Label:"):].strip()
            elif key_lower.startswith("range:"):
                current_channel["range"] = _parse_range_mv(
                    stripped[len("Range:"):].strip()
                )
            elif key_lower.startswith("low:"):
                current_channel["low"] = _parse_freq_hz(
                    stripped[len("Low:"):].strip()
                )
            elif key_lower.startswith("high:"):
                current_channel["high"] = _parse_freq_hz(
                    stripped[len("High:"):].strip()
                )
            elif key_lower.startswith("sample rate:"):
                current_channel["sample_rate"] = int(round(_parse_freq_hz(
                    stripped[len("Sample rate:"):].strip()
                )))
            # Color, Scale, File Type, Version, etc. are intentionally ignored
            # (MATLAB getbarddetails.m does the same).

        info["sample_rate"] = int(round(float(info["sample_rate"])))
        return info, channels, data_start_byte

    def _read_data_block(self, start_offset: int) -> np.ndarray:
        """Port of the data-loading half of ``loadBardFile.m``.

        Returns an int16 array shaped ``(NSamples, NChannels)``.
        """
        with self.file_name.open("rb") as fh:
            fh.seek(start_offset)
            raw = fh.read()
        # MATLAB: fData(fData == ',') = ' '; data = sscanf(fData, '%ld');
        text = raw.decode("ascii", errors="ignore").replace(",", " ")
        tokens = text.split()
        if not tokens:
            raise ValueError("BARDFILE/LOADBARDFILE: no data found after [Data]")

        # MATLAB: read as int64 (%ld), then int16(...) — narrowing matches
        # the documented A/D Converter output width (ADCBITS = 16).
        values64 = np.fromiter(
            (int(t) for t in tokens), dtype=np.int64, count=len(tokens)
        )
        values = values64.astype(np.int16, copy=False)

        if values.size % self.n_channels != 0:
            raise ValueError(
                "BARDFILE/LOADBARDFILE: wrong dimensions of data."
            )
        n_lines = values.size // self.n_channels

        # MATLAB: reshape(data, NChannels, nLines)' -> (nLines, NChannels)
        return values.reshape(n_lines, self.n_channels)

    def egm(
        self,
        i_time: Union[str, slice, int, Sequence[int], np.ndarray],
        i_channel: Union[str, Sequence[str], int, Sequence[int], np.ndarray],
    ) -> Optional[np.ndarray]:
        """Port of ``egm.m`` — return scaled electrogram(s) in mV.

        Parameters
        ----------
        i_time
            ``':'``, a ``slice``, an int, or an array-like of sample indices
            (0-based, unlike MATLAB).
        i_channel
            ``':'``, a channel label (str), a list of labels, an int, or an
            array-like of channel indices.

        Returns
        -------
        np.ndarray, shape ``(n_time, n_selected_channels)``, ``float64``, mV.
        Returns ``None`` if the data block has not been loaded.
        """
        if self.ch_data_file_map is None:
            return None

        if isinstance(i_channel, str):
            if i_channel == ":":
                idx = np.arange(self.n_channels)
            else:
                idx = self.ch_names_to_indices(i_channel)
        elif isinstance(i_channel, (list, tuple)) and all(
            isinstance(c, str) for c in i_channel
        ):
            idx = self.ch_names_to_indices(list(i_channel))
        else:
            idx = np.atleast_1d(np.asarray(i_channel, dtype=int))

        if isinstance(i_time, str) and i_time == ":":
            block = self.ch_data_file_map[:, idx]
        elif isinstance(i_time, (slice, int)):
            block = self.ch_data_file_map[i_time, idx]
            if block.ndim == 1:
                block = block[np.newaxis, :]
        else:
            t_idx = np.atleast_1d(np.asarray(i_time, dtype=int))
            block = self.ch_data_file_map[np.ix_(t_idx, idx)]

        # MATLAB: b = double(a2d) * 2 / 2^ADCBITS; b(:,i) = b(:,i) * ChRange(i)
        b = block.astype(np.float64) * (2.0 / (2 ** self.ADCBITS))
        b = b * self.ch_range[idx][np.newaxis, :]
        return b

    def ch_names_to_indices(
        self,
        ch_names: Union[str, Sequence[str]],
    ) -> np.ndarray:
        """Port of ``chNames2Indices.m``. Returns 0-based indices."""
        if isinstance(ch_names, str):
            if ch_names == ":":
                return np.arange(self.n_channels)
            ch_names = [ch_names]
        if not all(isinstance(n, str) for n in ch_names):
            raise TypeError("BARDFILE/FINDCHANNELINDEX: unrecognised format.")

        ind: List[int] = []
        for name in ch_names:
            matches = [i for i, n in enumerate(self.ch_name) if n == name]
            if len(matches) == 0:
                raise ValueError(
                    f"BARDFILE/SUBSREF: No match was found for label {name!r}."
                )
            if len(matches) > 1:
                raise ValueError(
                    f"BARDFILE/SUBSREF: More than one match was found for label {name!r}."
                )
            ind.append(matches[0])
        return np.asarray(ind, dtype=int)

    def __repr__(self) -> str:
        return (
            f"BardFile(file_name={self.short_file_name!r}, "
            f"n_channels={self.n_channels}, n_samples={self.n_samples}, "
            f"sample_rate={self.sample_rate})"
        )


def _new_channel() -> dict:
    return {
        "label": "",
        "range": 0.0,
        "low": 0.0,
        "high": 0.0,
        "sample_rate": 0,
    }


_FREQ_RE = re.compile(
    r"\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*(k?Hz)?\s*$",
    re.IGNORECASE,
)
_RANGE_RE = re.compile(
    r"\s*([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*(mv)?\s*$",
    re.IGNORECASE,
)


def _parse_freq_hz(value: str) -> float:
    """Parse '25Hz', '.05Hz', '1kHz', '0.5kHz', or a bare number -> Hz.

    Mirrors the MATLAB unit handling in ``getbarddetails.m`` for the
    ``Low:``, ``High:`` and ``Sample rate:`` fields, including the
    ``kHz``-detection branch::

        if textLine(end-2) == 'k'   % units are kHz
            high = 1000 * str2double(...)
    """
    m = _FREQ_RE.match(value)
    if not m:
        raise ValueError(f"BARDFILE: cannot parse frequency {value!r}")
    val = float(m.group(1))
    unit = (m.group(2) or "").lower()
    if unit == "khz":
        val *= 1000.0
    return val


def _parse_range_mv(value: str) -> float:
    """Parse '5mv', '5 mV', '5' -> numeric mV full-scale."""
    m = _RANGE_RE.match(value)
    if not m:
        raise ValueError(f"BARDFILE: cannot parse range {value!r}")
    return float(m.group(1))


if __name__ == "__main__":
    import sys

    default_path = (
        "/Users/vinush-vigneswaran/Documents/09_DATASETS/bard_data/"
        "HH2_bard_txt/csd450_preablation_locationonly.txt"
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    bf = Bard(path)
    print(bf.info)
    print("a2d dtype:", bf.ch_data_file_map.dtype)
    print("a2d shape (NSamples, NChannels):", bf.ch_data_file_map.shape)
    print("ChName:", bf.ch_name)
    print("ChRange (mV):", bf.ch_range)
    print("ChLow  (Hz):", bf.ch_low)
    print("ChHigh (Hz):", bf.ch_high)
    print("First 3 samples x all channels (mV):")
    print(bf.egm(slice(0, 3), ":"))
