from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

class ParseError(Exception):
    """Exception raised for errors during parsing."""

@dataclass
class Channel:
    channel_number: Optional[int] = None
    color: Optional[str] = None
    high: float = 0
    label: Optional[str] = None
    low: float = 0
    range: float = 0
    sample_rate: Optional[int] = None
    scale: Optional[int] = None


@dataclass
class EGM:
    path: Optional[Path] = None
    start_time: Any = None
    end_time: Any = None
    n_channels: Optional[int] = None
    n_samples: Optional[int] = None
    sample_rate_hz: Optional[int] = None
    channels: Dict[int, Channel] = field(default_factory=dict)
    data: Optional[np.ndarray] = None
    stimulus_indices: Any = None
    filter_band: List[int] = field(default_factory=lambda: [100, 500])

    @property
    def friendly_name(self):
        return self.path.name if self.path is not None else None

    @property
    def short_file_name(self):
        return self.path.stem if self.path is not None else None

    @property
    def info(self):
        return (
            f"Type\t\t: EGM\n"
            f"File Name\t\t: {self.short_file_name}\n"
            f"Start Time\t\t: {self.start_time}\n"
            f"End Time\t\t: {self.end_time}\n"
            f"N Channels\t\t: {self.n_channels}\n"
            f"N Samples\t\t: {self.n_samples}\n"
            f"Sample Rate Hz\t: {self.sample_rate_hz}\n"
            f"Path\t\t: {self.path}\n"
        )

