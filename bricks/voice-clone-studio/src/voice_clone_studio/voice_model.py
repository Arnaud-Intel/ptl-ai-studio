"""Model loading, OpenVINO conversion wrappers, and the enroll/synthesize
logic shared by both engine backends -- they run the identical checkpoints,
so the only thing that differs per engine is what executes the two heaviest
forward passes (`BaseSpeakerTTS.model.infer`, `ToneColorConverter.model.voice_conversion`).
"""
import tempfile
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from ._openvoice import se_extractor
from ._openvoice.api import BaseSpeakerTTS, OpenVoiceBaseClass, ToneColorConverter

REPO_ID = "myshell-ai/OpenVoice"
SAMPLE_RATE = 22050
STYLES = [
    "default", "whispering", "shouting", "excited", "cheerful",
    "terrified", "angry", "sad", "friendly",
]

_CHECKPOINT_FILES = {
    "en_config": "checkpoints/base_speakers/EN/config.json",
    "en_ckpt": "checkpoints/base_speakers/EN/checkpoint.pth",
    "en_default_se": "checkpoints/base_speakers/EN/en_default_se.pth",
    "conv_config": "checkpoints/converter/config.json",
    "conv_ckpt": "checkpoints/converter/checkpoint.pth",
}


def resolve_checkpoints() -> dict:
    return {key: hf_hub_download(REPO_ID, rel) for key, rel in _CHECKPOINT_FILES.items()}


def load_models():
    """Loads the base speaker TTS, the tone converter, and the TTS's own
    default-voice embedding (the fixed 'source' tone every clone starts
    from) -- pure PyTorch, CPU. Both engine backends load through this."""
    paths = resolve_checkpoints()

    tts = BaseSpeakerTTS(paths["en_config"], device="cpu")
    tts.load_ckpt(paths["en_ckpt"])

    converter = ToneColorConverter(paths["conv_config"], device="cpu")
    converter.load_ckpt(paths["conv_ckpt"])

    source_se = torch.load(paths["en_default_se"], map_location="cpu")
    return tts, converter, source_se


def load_tts_only() -> BaseSpeakerTTS:
    """Loads just BaseSpeakerTTS -- for consumers that only need speech in
    the base voice, with no cloning stage at all (e.g. voice-assistant's
    generic spoken-reply voice). Skips loading ToneColorConverter entirely,
    since it would otherwise sit there unused."""
    paths = resolve_checkpoints()
    tts = BaseSpeakerTTS(paths["en_config"], device="cpu")
    tts.load_ckpt(paths["en_ckpt"])
    return tts


def accelerate_tts_with_openvino(tts: BaseSpeakerTTS, device: str = "CPU") -> None:
    """Patches `tts.model.infer` in place to run via cached OpenVINO IR
    instead of native PyTorch -- the TTS half of what OpenVINOCloner does,
    for a caller (like voice-assistant) that never needs tone conversion."""
    from openvino import Core

    ir_path = ir_cache_dir() / "openvoice_en_tts.xml"
    wrapped = OVWrapTTS(tts)

    core = Core()
    if ir_path.exists():
        ov_model = core.read_model(ir_path)
    else:
        import openvino as ov

        ov_model = ov.convert_model(wrapped, example_input=wrapped.get_example_input())
        ov.save_model(ov_model, ir_path)

    compiled = core.compile_model(ov_model, device)

    def infer(x, x_lengths, sid, noise_scale, length_scale, noise_scale_w):
        output = compiled((x, x_lengths, sid, noise_scale, length_scale, noise_scale_w))
        return (torch.tensor(output[0]),)

    tts.model.infer = infer


def ir_cache_dir() -> Path:
    d = Path.home() / ".cache" / "panther-lake-ai-studio" / "voice-clone-studio" / "openvino_ir"
    d.mkdir(parents=True, exist_ok=True)
    return d


class _OVWrapBase(torch.nn.Module):
    """Both BaseSpeakerTTS and ToneColorConverter expose their real entry
    point as a custom method (`infer`/`voice_conversion`), not `forward` --
    OpenVINO's converter needs an actual nn.Module.forward to trace, so
    each subclass below wraps one method as `forward`."""

    def __init__(self, voice_model: OpenVoiceBaseClass):
        super().__init__()
        self.voice_model = voice_model
        for par in voice_model.model.parameters():
            par.requires_grad = False


class OVWrapTTS(_OVWrapBase):
    def get_example_input(self):
        stn_tst = self.voice_model.get_text("this is original text", self.voice_model.hps, False)
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
        speaker_id = torch.LongTensor([1])
        return (x_tst, x_tst_lengths, speaker_id, torch.tensor(0.667), torch.tensor(1.0), torch.tensor(0.6))

    def forward(self, x, x_lengths, sid, noise_scale, length_scale, noise_scale_w):
        return self.voice_model.model.infer(x, x_lengths, sid, noise_scale, length_scale, noise_scale_w)


class OVWrapConverter(_OVWrapBase):
    def get_example_input(self):
        y = torch.randn([1, 513, 238], dtype=torch.float32)
        y_lengths = torch.LongTensor([y.size(-1)])
        target_se = torch.randn(1, 256, 1)
        source_se = torch.randn(1, 256, 1)
        return (y, y_lengths, source_se, target_se, torch.tensor(0.3))

    def forward(self, y, y_lengths, sid_src, sid_tgt, tau):
        return self.voice_model.model.voice_conversion(y, y_lengths, sid_src, sid_tgt, tau)


def enroll(converter: ToneColorConverter, reference_audio_path: str):
    """Extracts a target speaker/tone embedding from a short reference
    clip -- zero-shot: this is inference over the clip, not a training
    step, so it takes seconds, not minutes."""
    target_se, _ = se_extractor.get_se(reference_audio_path, converter)
    return target_se


def synthesize(tts: BaseSpeakerTTS, converter: ToneColorConverter, source_se, text: str, target_se, style: str = "default", tau: float = 0.3):
    """Generates `text` in the base speaker's `style`, then converts its
    tone color to `target_se`. Returns (audio: np.ndarray float32, sample_rate)."""
    fd, base_path = tempfile.mkstemp(suffix=".wav")
    import os
    os.close(fd)
    try:
        tts.tts(text, base_path, speaker=style, language="English")
        audio = converter.convert(
            audio_src_path=base_path,
            src_se=source_se,
            tgt_se=target_se,
            output_path=None,
            tau=tau,
            message="PantherLakeAIStudio",
        )
    finally:
        Path(base_path).unlink(missing_ok=True)
    return audio, converter.hps.data.sampling_rate
