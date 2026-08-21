import torch
from openvino import Core

from . import voice_model


class OpenVINOCloner:
    """Same checkpoints as the portable engine, but the two heaviest
    forward passes -- BaseSpeakerTTS's phoneme-to-waveform synthesis and
    ToneColorConverter's tone transform -- run as OpenVINO IR models
    instead of native PyTorch, on CPU/GPU/NPU. Converted IR is cached on
    disk (voice_model.ir_cache_dir()) since conversion itself takes
    several seconds and only needs to happen once per machine."""

    def __init__(self, device="CPU", model_path=None):
        self.tts, self.converter, self.source_se = voice_model.load_models()

        cache_dir = voice_model.ir_cache_dir()
        tts_ir_path = cache_dir / "openvoice_en_tts.xml"
        conv_ir_path = cache_dir / "openvoice_tone_conversion.xml"

        core = Core()
        ov_tts = self._convert_or_load(core, tts_ir_path, voice_model.OVWrapTTS(self.tts))
        ov_conv = self._convert_or_load(core, conv_ir_path, voice_model.OVWrapConverter(self.converter))

        compiled_tts = core.compile_model(ov_tts, device)
        compiled_conv = core.compile_model(ov_conv, device)

        self.tts.model.infer = self._patched_infer(compiled_tts)
        self.converter.model.voice_conversion = self._patched_voice_conversion(compiled_conv)

    @staticmethod
    def _convert_or_load(core, ir_path, wrapped_model):
        import openvino as ov

        if ir_path.exists():
            return core.read_model(ir_path)
        converted = ov.convert_model(wrapped_model, example_input=wrapped_model.get_example_input())
        ov.save_model(converted, ir_path)
        return converted

    @staticmethod
    def _patched_infer(compiled_model):
        def infer(x, x_lengths, sid, noise_scale, length_scale, noise_scale_w):
            output = compiled_model((x, x_lengths, sid, noise_scale, length_scale, noise_scale_w))
            return (torch.tensor(output[0]),)
        return infer

    @staticmethod
    def _patched_voice_conversion(compiled_model):
        def voice_conversion(y, y_lengths, sid_src, sid_tgt, tau):
            output = compiled_model((y, y_lengths, sid_src, sid_tgt, tau))
            return (torch.tensor(output[0]),)
        return voice_conversion

    def enroll(self, reference_audio_path):
        return voice_model.enroll(self.converter, reference_audio_path)

    def synthesize(self, text, target_se, style="default", tau=0.3):
        return voice_model.synthesize(self.tts, self.converter, self.source_se, text, target_se, style, tau)
