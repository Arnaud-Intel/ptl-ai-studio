"""Trimmed from myshell-ai/OpenVoice's openvoice/text/cleaners.py (MIT License).

Upstream's `cjke_cleaners2` also handles [ZH]/[JA]/[KO]-tagged spans via
mandarin.py/japanese.py/korean.py -- three extra NLP dependency chains
(pypinyin, cn2an, jieba, ...) this brick has no use for, since every text
this brick ever hands to the cleaner is wrapped in `[EN]...[EN]` (see
`api.py`'s `BaseSpeakerTTS.tts`, English-only in this brick). Keeping just
the `[EN]` branch is behaviorally identical for that input, not a shortcut.
"""
import re

from .english import english_to_ipa2


def cjke_cleaners2(text):
    text = re.sub(r'\[EN\](.*?)\[EN\]',
                  lambda x: english_to_ipa2(x.group(1)) + ' ', text)
    text = re.sub(r'\s+$', '', text)
    text = re.sub(r'([^\.,!\?\-…~])$', r'\1.', text)
    return text
