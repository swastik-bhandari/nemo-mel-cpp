# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright (c) 2018 Ryan Leary
# (MIT-licensed portions adapted from https://github.com/ryanleary/patter)
#
# ============================================================================
# STANDALONE NeMo Mel-Spectrogram extractor  --  single file, zero NeMo deps.
#
# Everything below is copied VERBATIM from:
#   nemo/collections/asr/parts/preprocessing/features.py   (normalize_batch,
#       splice_frames, FilterbankFeatures)
#   nemo/collections/asr/parts/preprocessing/segment.py    (select_channels,
#       _convert_samples_to_float32, the soundfile loading path of
#       AudioSegment.from_file)
#   nemo/collections/asr/modules/audio_preprocessing.py    (the float32
#       casting contract of AudioToMelSpectrogramPreprocessor.forward and
#       its default arguments)
#
# The ONLY changes:
#   * `from nemo.utils import logging`  ->  stdlib `logging`
#   * AudioAugmentor / AudioSegment imports removed (used only by the
#     WaveformFeaturizer file-loading wrapper, never by the mel math);
#     their file-loading behaviour is reproduced exactly in
#     `load_audio()` below.
#
# Requires only:  torch, librosa, soundfile, numpy
# ============================================================================

import logging
import math
import random
from typing import Iterable, Optional, Union

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

CONSTANT = 1e-5


# ============================================================================
# From nemo/collections/asr/parts/preprocessing/features.py  (verbatim)
# ============================================================================


def normalize_batch(x, seq_len, normalize_type):
    x_mean = None
    x_std = None
    if normalize_type == "per_feature":
        batch_size = x.shape[0]
        max_time = x.shape[2]

        # When doing stream capture to a graph, item() is not allowed
        # becuase it calls cudaStreamSynchronize(). Therefore, we are
        # sacrificing some error checking when running with cuda graphs.
        if (
            torch.cuda.is_available()
            and not torch.cuda.is_current_stream_capturing()
            and torch.any(seq_len == 1).item()
        ):
            raise ValueError(
                "normalize_batch with `per_feature` normalize_type received a tensor of length 1. This will result "
                "in torch.std() returning nan. Make sure your audio length has enough samples for a single "
                "feature (ex. at least `hop_length` for Mel Spectrograms)."
            )
        time_steps = torch.arange(max_time, device=x.device).unsqueeze(0).expand(batch_size, max_time)
        valid_mask = time_steps < seq_len.unsqueeze(1)
        x_mean_numerator = torch.where(valid_mask.unsqueeze(1), x, 0.0).sum(axis=2)
        x_mean_denominator = valid_mask.sum(axis=1)
        x_mean = x_mean_numerator / x_mean_denominator.unsqueeze(1)

        # Subtract 1 in the denominator to correct for the bias.
        x_std = torch.sqrt(
            torch.sum(torch.where(valid_mask.unsqueeze(1), x - x_mean.unsqueeze(2), 0.0) ** 2, axis=2)
            / (x_mean_denominator.unsqueeze(1) - 1.0)
        )
        x_std = x_std.masked_fill(x_std.isnan(), 0.0)  # edge case: only 1 frame in denominator
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2), x_mean, x_std
    elif normalize_type == "all_features":
        x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
        for i in range(x.shape[0]):
            x_mean[i] = x[i, :, : seq_len[i].item()].mean()
            x_std[i] = x[i, :, : seq_len[i].item()].std()
        # make sure x_std is not zero
        x_std += CONSTANT
        return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1), x_mean, x_std
    elif "fixed_mean" in normalize_type and "fixed_std" in normalize_type:
        x_mean = torch.tensor(normalize_type["fixed_mean"], device=x.device)
        x_std = torch.tensor(normalize_type["fixed_std"], device=x.device)
        return (
            (x - x_mean.view(x.shape[0], x.shape[1]).unsqueeze(2)) / x_std.view(x.shape[0], x.shape[1]).unsqueeze(2),
            x_mean,
            x_std,
        )
    else:
        return x, x_mean, x_std


def splice_frames(x, frame_splicing):
    """Stacks frames together across feature dim

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim*frame_splicing, num_frames

    """
    seq = [x]
    for n in range(1, frame_splicing):
        seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
    return torch.cat(seq, dim=1)


class FilterbankFeatures(nn.Module):
    """Featurizer that converts wavs to Mel Spectrograms.
    See AudioToMelSpectrogramPreprocessor for args.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_window_size=320,
        n_window_stride=160,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        preemph=0.97,
        nfilt=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        log_zero_guard_type="add",
        log_zero_guard_value=2**-24,
        dither=CONSTANT,
        pad_to=16,
        max_duration=16.7,
        frame_splicing=1,
        exact_pad=False,
        pad_value=0,
        mag_power=2.0,
        use_grads=False,
        rng=None,
        nb_augmentation_prob=0.0,
        nb_max_freq=4000,
        mel_norm="slaney",
        stft_exact_pad=False,  # Deprecated arguments; kept for config compatibility
        stft_conv=False,  # Deprecated arguments; kept for config compatibility
    ):
        super().__init__()
        if stft_conv or stft_exact_pad:
            logging.warning(
                "Using torch_stft is deprecated and has been removed. The values have been forcibly set to False "
                "for FilterbankFeatures and AudioToMelSpectrogramPreprocessor. Please set exact_pad to True "
                "as needed."
            )
        if exact_pad and n_window_stride % 2 == 1:
            raise NotImplementedError(
                f"{self} received exact_pad == True, but hop_size was odd. If audio_length % hop_size == 0. Then the "
                "returned spectrogram would not be of length audio_length // hop_size. Please use an even hop_size."
            )
        self.log_zero_guard_value = log_zero_guard_value
        if (
            n_window_size is None
            or n_window_stride is None
            or not isinstance(n_window_size, int)
            or not isinstance(n_window_stride, int)
            or n_window_size <= 0
            or n_window_stride <= 0
        ):
            raise ValueError(
                f"{self} got an invalid value for either n_window_size or "
                f"n_window_stride. Both must be positive ints."
            )
        logging.info(f"PADDING: {pad_to}")

        self.sample_rate = sample_rate
        self.win_length = n_window_size
        self.hop_length = n_window_stride
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))
        self.stft_pad_amount = (self.n_fft - self.hop_length) // 2 if exact_pad else None
        self.exact_pad = exact_pad
        self.sample_rate = sample_rate

        if exact_pad:
            logging.info("STFT using exact pad")
        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length, periodic=False) if window_fn else None
        self.register_buffer("window", window_tensor)

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2

        filterbanks = torch.tensor(
            librosa.filters.mel(
                sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq, norm=mel_norm
            ),
            dtype=torch.float,
        ).unsqueeze(0)
        self.register_buffer("fb", filterbanks)

        # Calculate maximum sequence length
        max_length = self.get_seq_len(torch.tensor(max_duration * sample_rate, dtype=torch.float))
        max_pad = pad_to - (max_length % pad_to) if pad_to > 0 else 0
        self.max_length = max_length + max_pad
        self.pad_value = pad_value
        self.mag_power = mag_power

        # We want to avoid taking the log of zero
        # There are two options: either adding or clamping to a small value
        if log_zero_guard_type not in ["add", "clamp"]:
            raise ValueError(
                f"{self} received {log_zero_guard_type} for the "
                f"log_zero_guard_type parameter. It must be either 'add' or "
                f"'clamp'."
            )

        self.use_grads = use_grads
        if not use_grads:
            self.forward = torch.no_grad()(self.forward)
        self._rng = random.Random() if rng is None else rng
        self.nb_augmentation_prob = nb_augmentation_prob
        if self.nb_augmentation_prob > 0.0:
            if nb_max_freq >= sample_rate / 2:
                self.nb_augmentation_prob = 0.0
            else:
                self._nb_max_fft_bin = int((nb_max_freq / sample_rate) * n_fft)

        # log_zero_guard_value is the the small we want to use, we support
        # an actual number, or "tiny", or "eps"
        self.log_zero_guard_type = log_zero_guard_type
        logging.debug(f"sr: {sample_rate}")
        logging.debug(f"n_fft: {self.n_fft}")
        logging.debug(f"win_length: {self.win_length}")
        logging.debug(f"hop_length: {self.hop_length}")
        logging.debug(f"n_mels: {nfilt}")
        logging.debug(f"fmin: {lowfreq}")
        logging.debug(f"fmax: {highfreq}")
        logging.debug(f"using grads: {use_grads}")
        logging.debug(f"nb_augmentation_prob: {nb_augmentation_prob}")

    def stft(self, x):
        return torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False if self.exact_pad else True,
            window=self.window.to(dtype=torch.float, device=x.device),
            return_complex=True,
            pad_mode="constant",
        )

    def log_zero_guard_value_fn(self, x):
        if isinstance(self.log_zero_guard_value, str):
            if self.log_zero_guard_value == "tiny":
                return torch.finfo(x.dtype).tiny
            elif self.log_zero_guard_value == "eps":
                return torch.finfo(x.dtype).eps
            else:
                raise ValueError(
                    f"{self} received {self.log_zero_guard_value} for the "
                    f"log_zero_guard_type parameter. It must be either a "
                    f"number, 'tiny', or 'eps'"
                )
        else:
            return self.log_zero_guard_value

    def get_seq_len(self, seq_len):
        # Assuming that center is True is stft_pad_amount = 0
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length)
        return seq_len.to(dtype=torch.long)

    @property
    def filter_banks(self):
        return self.fb

    def forward(self, x, seq_len, linear_spec=False):
        seq_len_time = seq_len
        seq_len_unfixed = self.get_seq_len(seq_len)
        # fix for seq_len = 0 for streaming; if size was 0, it is always padded to 1, and normalizer fails
        seq_len = torch.where(seq_len == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed)

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant"
            ).squeeze(1)

        # dither (only in training mode for eval determinism)
        if self.training and self.dither > 0:
            x += self.dither * torch.randn_like(x)

        # do preemphasis
        if self.preemph is not None:
            timemask = torch.arange(x.shape[1], device=x.device).unsqueeze(0) < seq_len_time.unsqueeze(1)
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)
            x = x.masked_fill(~timemask, 0.0)

        # disable autocast to get full range of stft values
        with torch.amp.autocast(x.device.type, enabled=False):
            x = self.stft(x)

        # torch stft returns complex tensor (of shape [B,N,T]); so convert to magnitude
        # guard is needed for sqrt if grads are passed through
        guard = 0 if not self.use_grads else CONSTANT
        x = torch.view_as_real(x)
        x = torch.sqrt(x.pow(2).sum(-1) + guard)

        if self.training and self.nb_augmentation_prob > 0.0:
            for idx in range(x.shape[0]):
                if self._rng.random() < self.nb_augmentation_prob:
                    x[idx, self._nb_max_fft_bin :, :] = 0.0

        # get power spectrum
        if self.mag_power != 1.0:
            x = x.pow(self.mag_power)

        # return plain spectrogram if required
        if linear_spec:
            return x, seq_len

        # disable autocast, otherwise it might be automatically casted to fp16
        # on fp16 compatible GPUs and get NaN values for input value of 65520
        with torch.amp.autocast(x.device.type, enabled=False):
            # dot with filterbank energies
            x = torch.matmul(self.fb.to(x.dtype), x)
        # log features if required
        if self.log:
            if self.log_zero_guard_type == "add":
                x = torch.log(x + self.log_zero_guard_value_fn(x))
            elif self.log_zero_guard_type == "clamp":
                x = torch.log(torch.clamp(x, min=self.log_zero_guard_value_fn(x)))
            else:
                raise ValueError("log_zero_guard_type was not understood")

        # frame splicing if required
        if self.frame_splicing > 1:
            x = splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x, _, _ = normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len, device=x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).type(torch.bool).to(device=x.device), self.pad_value)
        del mask
        pad_to = self.pad_to
        if pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        elif pad_to > 0:
            pad_amt = x.size(-1) % pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, pad_to - pad_amt), value=self.pad_value)
        return x, seq_len


# ============================================================================
# From nemo/collections/asr/parts/preprocessing/segment.py  (verbatim)
# select_channels + _convert_samples_to_float32 + the soundfile branch of
# AudioSegment.from_file, reproduced as a function.
# ============================================================================


def select_channels(signal: np.ndarray, channel_selector=None) -> np.ndarray:
    """
    Convert a multi-channel signal to a single-channel signal by averaging over channels or
    selecting a single channel, or pass-through multi-channel signal when channel_selector is `None`.

    Args:
        signal: numpy array with shape (..., num_channels)
        channel selector: string denoting the downmix mode, an integer denoting the channel to be selected,
                          or an iterable of integers denoting a subset of channels. Channel selector is
                          using zero-based indexing. If set to `None`, the original signal will be returned.
                          Uses zero-based indexing.

    Returns:
        numpy array
    """
    if signal.ndim == 1:
        # For one-dimensional input, return the input signal.
        if channel_selector not in [None, 0, 'average']:
            raise ValueError(
                'Input signal is one-dimensional, channel selector (%s) cannot not be used.', str(channel_selector)
            )
        return signal

    num_channels = signal.shape[-1]
    num_samples = signal.size // num_channels  # handle multi-dimensional signals

    if num_channels >= num_samples:
        logging.warning(
            'Number of channels (%d) is greater or equal than number of samples (%d). '
            'Check for possible transposition.',
            num_channels,
            num_samples,
        )

    # Samples are arranged as (num_channels, ...)
    if channel_selector is None:
        # keep the original multi-channel signal
        pass
    elif channel_selector == 'average':
        # default behavior: downmix by averaging across channels
        signal = np.mean(signal, axis=-1)
    elif isinstance(channel_selector, int):
        # select a single channel
        if channel_selector >= num_channels:
            raise ValueError(f'Cannot select channel {channel_selector} from a signal with {num_channels} channels.')
        signal = signal[..., channel_selector]
    elif isinstance(channel_selector, Iterable):
        # select multiple channels
        if max(channel_selector) >= num_channels:
            raise ValueError(
                f'Cannot select channel subset {channel_selector} from a signal with {num_channels} channels.'
            )
        signal = signal[..., channel_selector]
        # squeeze the channel dimension if a single-channel is selected
        # this is done to have the same shape as when using integer indexing
        if len(channel_selector) == 1:
            signal = np.squeeze(signal, axis=-1)
    else:
        raise ValueError(f'Unexpected value for channel_selector ({channel_selector})')

    return signal


def _convert_samples_to_float32(samples):
    """Convert sample type to float32.
    Audio sample type is usually integer or float-point.
    Integers will be scaled to [-1, 1] in float32.
    """
    float32_samples = samples.astype('float32')
    if samples.dtype in (np.int8, np.int16, np.int32, np.int64):
        bits = np.iinfo(samples.dtype).bits
        float32_samples *= 1.0 / 2 ** (bits - 1)
    elif samples.dtype in (np.float16, np.float32, np.float64):
        pass
    else:
        raise TypeError("Unsupported sample type: %s." % samples.dtype)
    return float32_samples


def load_audio(
    audio_file: str,
    target_sr: int = 16000,
    int_values: bool = False,
    offset: float = 0,
    duration: float = 0,
    channel_selector=None,
) -> torch.Tensor:
    """
    Replicates exactly what happens in normal NeMo execution when a .wav file
    is loaded for the AudioToMelSpectrogramPreprocessor:

        WaveformFeaturizer.process(file_path)
          -> AudioSegment.from_file(file_path, target_sr=sample_rate)
               [soundfile branch: sf.SoundFile.read(dtype='float32')]
          -> AudioSegment.__init__:
               _convert_samples_to_float32 -> select_channels -> resample
          -> WaveformFeaturizer.process_segment:
               torch.tensor(segment.samples, dtype=torch.float)

    (The default AudioAugmentor is an empty augmentor - `perturb()` is a
    no-op - so it is omitted with zero effect on the output.)

    Returns a 1-D float32 torch.Tensor of samples at `target_sr`.
    """
    # --- soundfile branch of AudioSegment.from_file (verbatim logic) ---
    with sf.SoundFile(audio_file, 'r') as f:
        dtype = 'int32' if int_values else 'float32'
        sample_rate = f.samplerate
        if offset is not None and offset > 0:
            f.seek(int(offset * sample_rate))
        if duration is not None and duration > 0:
            samples = f.read(int(duration * sample_rate), dtype=dtype)
        else:
            samples = f.read(dtype=dtype)

    # --- AudioSegment.__init__ (verbatim logic) ---
    samples = _convert_samples_to_float32(samples)

    if samples.ndim == 1 and channel_selector not in [None, 0, 'average']:
        raise ValueError(
            'Input signal is one-dimensional, channel selector (%s) cannot not be used.', str(channel_selector)
        )
    elif samples.ndim == 2:
        samples = select_channels(samples, channel_selector)
    elif samples.ndim >= 3:
        raise NotImplementedError(
            'Signals with more than two dimensions (sample, channel) are currently not supported.'
        )

    if target_sr is not None and target_sr != sample_rate:
        # resample along the temporal dimension (axis=0) will be in librosa 0.10.0 (#1561)
        samples = samples.transpose()
        samples = librosa.core.resample(samples, orig_sr=sample_rate, target_sr=target_sr)
        samples = samples.transpose()

    # --- WaveformFeaturizer.process_segment (verbatim logic) ---
    return torch.tensor(samples, dtype=torch.float)


# ============================================================================
# Convenience wrapper == AudioToMelSpectrogramPreprocessor default pipeline.
#
# AudioToMelSpectrogramPreprocessor.forward does exactly:
#   1. cast input to float32          (input already float32 here)
#   2. featurizer(input_signal, length)   in torch.no_grad()
#   3. cast output to buffer dtype    (float32 by default -> no-op)
# Its default args map to the FilterbankFeatures defaults used below
# (window_size 0.02s -> 320 samples, window_stride 0.01s -> 160 samples,
#  features=64, dither=1e-5, pad_to=16, normalize='per_feature', ...).
# ============================================================================


def make_mel_featurizer(
    sample_rate=16000,
    window_size=0.02,
    window_stride=0.01,
    window="hann",
    normalize="per_feature",
    n_fft=None,
    preemph=0.97,
    features=64,
    lowfreq=0,
    highfreq=None,
    log=True,
    log_zero_guard_type="add",
    log_zero_guard_value=2**-24,
    dither=1e-5,
    pad_to=16,
    frame_splicing=1,
    exact_pad=False,
    pad_value=0,
    mag_power=2.0,
    mel_norm="slaney",
) -> FilterbankFeatures:
    """Builds FilterbankFeatures with the exact same argument mapping that
    AudioToMelSpectrogramPreprocessor performs (seconds -> samples)."""
    featurizer = FilterbankFeatures(
        sample_rate=sample_rate,
        n_window_size=int(window_size * sample_rate),
        n_window_stride=int(window_stride * sample_rate),
        window=window,
        normalize=normalize,
        n_fft=n_fft,
        preemph=preemph,
        nfilt=features,
        lowfreq=lowfreq,
        highfreq=highfreq,
        log=log,
        log_zero_guard_type=log_zero_guard_type,
        log_zero_guard_value=log_zero_guard_value,
        dither=dither,
        pad_to=pad_to,
        frame_splicing=frame_splicing,
        exact_pad=exact_pad,
        pad_value=pad_value,
        mag_power=mag_power,
        mel_norm=mel_norm,
    )
    # eval() => self.training False => dither branch skipped, exactly like
    # NeMo inference (deterministic output).
    featurizer.eval()
    return featurizer


def wav_to_mel(
    audio_file: str,
    featurizer: Optional[FilterbankFeatures] = None,
    device: Union[str, torch.device] = "cpu",
    channel_selector=None,
):
    """
    .wav (or flac/ogg) file  ->  NeMo mel spectrogram.

    Returns:
        mel:     torch.Tensor [1, n_mels, T]  (float32)
        mel_len: torch.Tensor [1]             (long, valid frames before padding)
    """
    if featurizer is None:
        featurizer = make_mel_featurizer()
    featurizer = featurizer.to(device).eval()

    samples = load_audio(audio_file, target_sr=featurizer.sample_rate, channel_selector=channel_selector)
    input_signal = samples.unsqueeze(0).to(device)                      # [1, T]
    length = torch.tensor([samples.shape[0]], dtype=torch.long, device=device)  # [1]

    with torch.no_grad():
        mel, mel_len = featurizer(input_signal.to(torch.float32), length)
    return mel, mel_len


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "example.wav"
    import soundfile as _sf
    _ch = _sf.info(path).channels
    cs = 'average' if _ch > 1 else None
    if _ch > 1:
        print(f"note: {path} has {_ch} channels -> downmixing with channel_selector='average'")
    mel, mel_len = wav_to_mel(path, channel_selector=cs)
    print(f"file:      {path}")
    print(f"mel shape: {tuple(mel.shape)}  (batch, n_mels, frames incl. pad_to-16 padding)")
    print(f"valid frames: {mel_len.item()}")
    print(f"dtype: {mel.dtype}, min={mel.min():.4f}, max={mel.max():.4f}, mean={mel.mean():.4f}")
