"""High-level reader that loads EGM files into the unified :class:`EGM` model.

The vendor-specific parsing (e.g. :class:`Bard`) stays in its own module —
this reader is the thin adaptor that turns a vendor object into an
:class:`EGM` populated from :mod:`openep.data_structures.egm`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from openep.data_structures.egm import EGM, Channel
from openep.io.egm_readers.bard import Bard


class EGMReader:
    """Read an EGM file and return a populated :class:`EGM`.

    Currently dispatches to :class:`Bard` for all inputs. Additional vendor
    readers can be added later behind the same interface.
    """

    @staticmethod
    def load_bard(file_path: Union[str, Path]) -> EGM:
        bard = Bard(file_path)
        return EGMReader._from_bard(bard)

    @staticmethod
    def _from_bard(bard: Bard) -> EGM:
        channels = {
            i: Channel(
                channel_number=i,
                label=bard.ch_name[i],
                range=float(bard.ch_range[i]),
                low=float(bard.ch_low[i]),
                high=float(bard.ch_high[i]),
                sample_rate=bard.sample_rate,
            )
            for i in range(bard.n_channels)
        }

        data = bard.egm(":", ":")

        return EGM(
            path=bard.file_name,
            start_time=bard.start_time,
            end_time=bard.end_time,
            n_channels=bard.n_channels,
            n_samples=bard.n_samples,
            sample_rate_hz=bard.sample_rate,
            channels=channels,
            data=data,
            stimulus_indices=bard._stim_indices,
            filter_band=list(bard._filter),
        )


if __name__ == "__main__":
    path = ("test/path.txt")
    egm = EGMReader.load_bard(path)
    print(egm.info)
    print("data dtype:", egm.data.dtype)
    print("data shape (NSamples, NChannels):", egm.data.shape)
    print("channel 0:", egm.channels[0])
