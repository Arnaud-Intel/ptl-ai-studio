"""Trimmed from myshell-ai/OpenVoice's openvoice/se_extractor.py (MIT License).

Upstream offers two ways to split a reference clip into non-silent chunks
before averaging a speaker embedding across them: Silero VAD, or a
faster-whisper transcription pass used purely for its word timestamps.
This brick only ever needs the former (it's what Intel's own OpenVoice ->
OpenVINO conversion notebook uses too), so the whisper path -- and the
faster-whisper/whisper-timestamped dependencies it pulls in -- is dropped
entirely. The VAD model itself is also loaded via torch.hub's current,
un-pinned entrypoint instead of upstream's manual pinned-zip workaround for
an old breaking change, since that workaround is no longer needed against
the current silero-vad release.
"""
import base64
import hashlib
import os
from glob import glob

import librosa
import numpy as np
import torch
from pydub import AudioSegment

_vad_model = None
_vad_utils = None


def _load_vad():
    global _vad_model, _vad_utils
    if _vad_model is None:
        _vad_model, _vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
        )
    return _vad_model, _vad_utils


def split_audio_vad(audio_path, audio_name, target_dir, split_seconds=10.0):
    model, utils = _load_vad()
    get_speech_timestamps = utils[0]

    # Silero's own `read_audio` helper goes through torchaudio's I/O
    # backends, which (as of the torchaudio version paired with our torch
    # pin) require the separate `torchcodec` package to actually decode
    # anything -- an extra dependency with no other purpose here. librosa
    # (already a dependency for spectrogram extraction elsewhere in this
    # package) reads the same file into an equivalent mono float array.
    sample_rate = 16000
    array, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
    wav = torch.from_numpy(array)
    segments = get_speech_timestamps(wav, model, sampling_rate=sample_rate)
    segments = [(seg["start"] / sample_rate, seg["end"] / sample_rate) for seg in segments]

    audio_active = AudioSegment.silent(duration=0)
    audio = AudioSegment.from_file(audio_path)
    for start_time, end_time in segments:
        audio_active += audio[int(start_time * 1000): int(end_time * 1000)]

    audio_dur = audio_active.duration_seconds
    target_folder = os.path.join(target_dir, audio_name)
    wavs_folder = os.path.join(target_folder, "wavs")
    os.makedirs(wavs_folder, exist_ok=True)

    num_splits = max(1, int(np.round(audio_dur / split_seconds)))
    assert audio_dur > 0, "no speech detected in the reference clip -- try a clearer recording"
    interval = audio_dur / num_splits

    start_time = 0.0
    for i in range(num_splits):
        end_time = audio_dur if i == num_splits - 1 else min(start_time + interval, audio_dur)
        output_file = f"{wavs_folder}/{audio_name}_seg{i}.wav"
        audio_active[int(start_time * 1000): int(end_time * 1000)].export(output_file, format="wav")
        start_time = end_time

    return wavs_folder


def _hash_numpy_array(audio_path):
    array, _ = librosa.load(audio_path, sr=None, mono=True)
    digest = hashlib.sha256(array.tobytes()).digest()
    return base64.b64encode(digest).decode("utf-8")[:16].replace("/", "_^")


def get_se(audio_path, vc_model, target_dir="processed"):
    audio_name = f"{os.path.basename(audio_path).rsplit('.', 1)[0]}_{_hash_numpy_array(audio_path)}"
    wavs_folder = split_audio_vad(audio_path, target_dir=target_dir, audio_name=audio_name)

    audio_segs = glob(f"{wavs_folder}/*.wav")
    if len(audio_segs) == 0:
        raise RuntimeError("No speech segments found in the reference clip.")

    return vc_model.extract_se(audio_segs), audio_name
