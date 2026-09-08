from . import voice_model


class PortableCloner:
    """Plain PyTorch, CPU only -- the same checkpoints as the openvino
    engine, run through their native torch forward passes untouched."""

    def __init__(self, model_path=None, on_downloading=None):
        self.tts, self.converter, self.source_se = voice_model.load_models(
            local_dir=model_path, on_downloading=on_downloading
        )

    def enroll(self, reference_audio_path):
        return voice_model.enroll(self.converter, reference_audio_path)

    def synthesize(self, text, target_se, style="default", tau=0.3):
        return voice_model.synthesize(self.tts, self.converter, self.source_se, text, target_se, style, tau)
