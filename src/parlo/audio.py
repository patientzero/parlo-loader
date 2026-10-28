"""Audio is decoded only by load_audio."""
from dataclasses import dataclass
from math import gcd
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray
    sample_rate: int


def load_audio(path, *, start_ms=None, end_ms=None, mono=False, sample_rate=None):
    """Return float32 samples (frames, channels), or (frames,) with mono=True.

    Millisecond boundaries select [floor(start*sr/1000), floor(end*sr/1000)).
    Never clamp, pad, or repair an invalid annotation.
    """
    with sf.SoundFile(path) as stream:
        native = stream.samplerate
        duration = len(stream) * 1000 / native
        for value in (start_ms, end_ms):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError('Audio boundaries must be integer milliseconds or None')
        start = 0 if start_ms is None else start_ms
        end = duration if end_ms is None else end_ms
        if not 0 <= start < end <= duration:
            raise ValueError(f'Invalid audio boundaries {start_ms}, {end_ms}; duration is {duration:.3f} ms')
        first = start * native // 1000
        last = len(stream) if end_ms is None else end * native // 1000
        if last <= first:
            raise ValueError('Excerpt contains no samples')
        stream.seek(first)
        data = stream.read(last - first, dtype='float32', always_2d=True)
    if mono:
        data = data.mean(axis=1)
    if sample_rate is not None:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise ValueError('sample_rate must be a positive integer')
        if sample_rate != native:
            from scipy.signal import resample_poly
            divisor = gcd(native, sample_rate)
            data = resample_poly(data, sample_rate // divisor, native // divisor, axis=0).astype('float32')
            native = sample_rate
    return Audio(data, native)
