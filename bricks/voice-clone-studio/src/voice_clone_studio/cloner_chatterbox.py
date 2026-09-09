from . import chatterbox_model


class ChatterboxCloner:
    """Chatterbox-Turbo through ONNX Runtime, on the CPU.

    No delivery styles and no tone-conversion strength: this model takes
    its delivery from the reference clip itself rather than from a menu of
    base voices, so there is nothing to dial. What it does take is
    paralinguistic tags written into the text -- `[laugh]`, `[sigh]` -- and
    those are native to the model, not post-processing.
    """

    supports_styles = False
    supports_tags = True
    sample_rate = chatterbox_model.SAMPLE_RATE

    def __init__(self, model_path=None, on_downloading=None):
        self.model = chatterbox_model.ChatterboxModel(model_path=model_path, on_downloading=on_downloading)

    def enroll(self, reference_audio_path):
        return self.model.enroll(reference_audio_path)

    def synthesize(self, text, target_se):
        return self.model.synthesize(text, target_se)
