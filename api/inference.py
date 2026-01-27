import os
import sys
import torch
import json
import re
import io
import logging
import base64
import numpy as np
from scipy.io import wavfile
from typing import Dict, Optional, List, Tuple

from api.config import settings

# Helper for resolving paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ======================================================
# AUDIO PATCH (Bypass torchcodec check for PyTorch > 2.9)
# ======================================================
import soundfile as sf
import types

def patched_load_audio(file_path):
    data, sr = sf.read(file_path, dtype='float32')
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.transpose(0, 1)
    return tensor, sr

# Mock torchcodec so import succeeds
fake_torchcodec = types.ModuleType("torchcodec")
from importlib.machinery import ModuleSpec
fake_torchcodec.__spec__ = ModuleSpec(name="torchcodec", loader=None)
sys.modules["torchcodec"] = fake_torchcodec
sys.modules["torchcodec.decoders"] = types.ModuleType("torchcodec.decoders")

# Imports from Coqui TTS
import TTS.tts.models.vits as vits_module
# Patch load_audio before it's used if possible, or just the module availability
# The error comes from TTS/__init__.py checking for torchcodec
# Just sys.modules patch above might be enough if done before ANY TTS import

from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.text import cleaners
from TTS.utils.audio import AudioProcessor

# Apply patch to vits module just in case
vits_module.load_audio = patched_load_audio

from tagger import Tagger

logger = logging.getLogger(__name__)

# ======================================================
# Custom Classes from test_model.py
# ======================================================

class GraphemeToPhonemeConverter:
    def __init__(self):
        self.mapping = {
            "ng'": "ŋ", "ng": "ŋg", "ny": "ɲ", "th": "θ", "dh": "ð",
            "ch": "tʃ", "sh": "ʃ", "j": "ɟ", "y": "j",
            "a": "a", "e": "ɛ", "i": "i", "o": "ɔ", "u": "u",
            "b": "b", "p": "p", "m": "m", "w": "w", "f": "f",
            "v": "v", "t": "t", "d": "d", "s": "s", "n": "n",
            "l": "l", "r": "ɾ", "k": "k", 
            "g": "g", 
            "h": "h"
        }
        pattern = "|".join(re.escape(k) for k in sorted(self.mapping.keys(), key=len, reverse=True))
        self.regex = re.compile(pattern)

    def predict(self, word):
        return self.regex.sub(lambda m: self.mapping[m.group(0)], word.lower().strip())

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

class DholuoVits(Vits):
    def __init__(self, config):
        super().__init__(config)
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            config.phoneme_language = "en"
            res = TTSTokenizer.init_from_config(config)
            self.tokenizer = res[0] if isinstance(res, tuple) else res
            config.phoneme_language = None
        self.tokenizer.text_cleaner = getattr(cleaners, "basic_cleaners")
        self.tokenizer.use_phonemes = False

# ======================================================
# Inference Engine
# ======================================================

class TTSInference:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TTSInference, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        logger.info("Initializing Dholuo TTS Inference Engine...")
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Initialize NLP Components
        try:
            self.tagger = Tagger(model_path=os.path.join(PROJECT_ROOT, "models/luo-pos"))
            self.g2p_engine = GraphemeToPhonemeConverter()
            lexicon_path = os.path.join(PROJECT_ROOT, "data/dholuo_lexicon.json")
            if os.path.exists(lexicon_path):
                with open(lexicon_path, "r", encoding="utf-8") as f:
                    self.lexicon = json.load(f)
            else:
                logger.warning(f"Lexicon not found at {lexicon_path}. Using fallback G2P only.")
                self.lexicon = {}
            logger.info("NLP components initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize NLP components: {e}")
            raise e

        # Initialize VITS Model
        try:
            model_path_abs = os.path.join(PROJECT_ROOT, settings.MODEL_PATH)
            config_path_abs = os.path.join(PROJECT_ROOT, settings.CONFIG_PATH)

            logger.info(f"Loading Model from {model_path_abs}")
            
            # Hardcoded characters from training script
            train_characters = "abcdefghijklmnopqrstuvwxyzɛɔɟɲŋθðɾtʃdʒ˩˥"
            
            self.config = VitsConfig(
                audio=None,
                run_eval=True,
                text_cleaner="basic_cleaners",
                use_phonemes=False,
                characters=CharactersConfig(
                    pad="_", eos="~", bos="^",
                    characters=train_characters,
                    punctuations=".,!?- ",
                )
            )
            # Load config json
            self.config.load_json(config_path_abs)
            
            # Init AudioProcessor
            self.ap = AudioProcessor(**self.config.audio)
            
            # Init Model
            self.model = DholuoVits(self.config)
            
            # Load Weights
            checkpoint = torch.load(model_path_abs, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            self.model.to(self.device).eval()
            
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load VITS model: {e}")
            raise e

        self._initialized = True

    def text_to_ipa(self, text: str) -> str:
        """Convert text to IPA using Tagger + Lexicon + G2P Fallback."""
        tagged_pairs = self.tagger.tag(text)
        ipa_list = []
        for word, tag in tagged_pairs:
            token_key = f"{word}_{tag}"
            if token_key in self.lexicon:
                # Ensure 'ɡ' -> 'g' fix
                ipa = self.lexicon[token_key].replace("ɡ", "g")
            else:
                ipa = self.g2p_engine.predict(word)
                if tag == "V": ipa += "˥"
                elif tag == "NN": ipa += "˩"
            ipa_list.append(ipa)
        return " ".join(ipa_list)

    def synthesize(
        self, 
        text: str, 
        speed: float = 1.0, 
        return_ipa: bool = False
    ) -> Dict:
        """Synthesize text to audio bytes."""
        # 1. Phonemize
        ipa_string = self.text_to_ipa(text)
        
        # 2. Prepare tokens
        seq = self.model.tokenizer.encode(ipa_string)
        if self.config.add_blank:
            seq = intersperse(seq, 0)
            
        t_in = torch.LongTensor(seq).unsqueeze(0).to(self.device)
        t_len = torch.LongTensor([t_in.size(1)]).to(self.device)
        
        # 3. Inference
        with torch.no_grad():
            outputs = self.model.inference(t_in, aux_input={"x_lengths": t_len})
        
        if isinstance(outputs, dict):
            wav = outputs["model_outputs"][0].data.cpu().numpy()
        else:
            wav = outputs[0].data.cpu().numpy()
            
        wav = wav.flatten()
        # Normalize
        wav = wav / (np.max(np.abs(wav)) + 1e-6)
        wav_16 = (wav * 32767).astype(np.int16)
        
        # Write to memory
        sr = self.config.audio.sample_rate
        buffer = io.BytesIO()
        wavfile.write(buffer, sr, wav_16)
        buffer.seek(0)
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        duration = len(wav) / sr
        
        result = {
            "audio": audio_base64,
            "duration": duration,
            "sample_rate": sr
        }
        
        if return_ipa:
            result["ipa_text"] = ipa_string
            
        return result

    def get_tagging(self, text: str) -> List[Tuple[str, str]]:
        return self.tagger.tag(text)

inference_engine = TTSInference()
