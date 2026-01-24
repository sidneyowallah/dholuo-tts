import os
import torch
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.text import cleaners

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
POS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/luo-pos")
TTS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/luo-tts/v13/checkpoint_130000.pth")
TTS_CONFIG_PATH = os.path.join(PROJECT_ROOT, "models/luo-tts/v13/config.json")
LEXICON_PATH = os.path.join(PROJECT_ROOT, "data/dholuo_lexicon.json")
AUDIO_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/audio/output")

def get_device():
    """Detect available device"""
    return "mps" if torch.backends.mps.is_available() else "cpu"

def intersperse(lst, item):
    """Insert item between every element in list"""
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

class DholuoVits(Vits):
    """Custom VITS model with tokenizer initialization"""
    def __init__(self, config):
        super().__init__(config)
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            config.phoneme_language = "en"
            res = TTSTokenizer.init_from_config(config)
            self.tokenizer = res[0] if isinstance(res, tuple) else res
            config.phoneme_language = None
        self.tokenizer.text_cleaner = getattr(cleaners, "basic_cleaners")
        self.tokenizer.use_phonemes = False
