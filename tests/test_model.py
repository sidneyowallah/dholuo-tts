import os
import sys
import json
import torch
import string
import re
import numpy as np
from scipy.io import wavfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Coqui specific imports
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.text import cleaners
from TTS.utils.audio import AudioProcessor

# Import your custom classes
from tagger import Tagger


class GraphemeToPhonemeConverter:
    def __init__(self):
        self.mapping = {
            "ng'": "ŋ", "ng": "ŋg", "ny": "ɲ", "th": "θ", "dh": "ð",
            "ch": "tʃ", "sh": "ʃ", "j": "ɟ", "y": "j",
            "a": "a", "e": "ɛ", "i": "i", "o": "ɔ", "u": "u",
            "b": "b", "p": "p", "m": "m", "w": "w", "f": "f",
            "v": "v", "t": "t", "d": "d", "s": "s", "n": "n",
            "l": "l", "r": "ɾ", "k": "k", 
            "g": "g", # <--- FIXED: Use standard 'g' (Matches your training config)
            "h": "h"
        }
        pattern = "|".join(re.escape(k) for k in sorted(self.mapping.keys(), key=len, reverse=True))
        self.regex = re.compile(pattern)

    def predict(self, word):
        return self.regex.sub(lambda m: self.mapping[m.group(0)], word.lower().strip())

# ==========================================
# 2. CONFIGURATION & PATHS
# ==========================================
CHECKPOINT_NAME = "checkpoint_150000"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POS_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/luo-pos")
LEXICON_PATH = os.path.join(PROJECT_ROOT, "data/dholuo_lexicon.json")

# ==========================================
# 3. MODEL CLASSES (Matching Train Script)
# ==========================================
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

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

class DholuoTTS:
    def __init__(self, gender="female"):
        print("🚀 Initializing POS-Aware Dholuo TTS...")
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.gender = gender
        
        TTS_MODEL_PATH = os.path.join(PROJECT_ROOT, f"models/luo-tts/{gender}/{CHECKPOINT_NAME}.pth")
        TTS_CONFIG_PATH = os.path.join(PROJECT_ROOT, f"models/luo-tts/{gender}/config.json")
        self.output_path = os.path.join(PROJECT_ROOT, f"tests/output/{CHECKPOINT_NAME}_{gender}_audio.wav")
        
        # Initialize NLP
        self.tagger = Tagger(model_path=POS_MODEL_PATH)
        self.g2p_engine = GraphemeToPhonemeConverter()
        with open(LEXICON_PATH, "r", encoding="utf-8") as f:
            self.lexicon = json.load(f)
        
        # EXACT COPIED STRINGS FROM YOUR TRAIN_VITS.PY
        # Do not modify these; they must match the training script exactly.
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
        self.config.load_json(TTS_CONFIG_PATH)
        
        self.ap = AudioProcessor(**self.config.audio)
        self.model = DholuoVits(self.config)
        
        checkpoint = torch.load(TTS_MODEL_PATH, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device).eval()
        print("✅ Pipeline Synchronized and Ready!")

    def text_to_ipa(self, text):
        # Uses your integrated Tagger + Morphological logic from Tagger class
        tagged_pairs = self.tagger.tag(text)
        ipa_list = []
        for word, tag in tagged_pairs:
            token_key = f"{word}_{tag}"
            if token_key in self.lexicon:
                # Lexicon cleanup: ensure any script 'ɡ' in lexicon is standard 'g'
                ipa = self.lexicon[token_key].replace("ɡ", "g")
            else:
                ipa = self.g2p_engine.predict(word)
                if tag == "V": ipa += "˥"
                elif tag == "NN": ipa += "˩"
            ipa_list.append(ipa)
        return " ".join(ipa_list)

    def speak(self, text, output_file=None):
        if output_file is None:
            output_file = self.output_path
        print(f"Input Text: {text}")
        ipa_string = self.text_to_ipa(text)
        print(f"Synthesizing: {ipa_string}")
        
        seq = self.model.tokenizer.encode(ipa_string)
        if self.config.add_blank:
            seq = intersperse(seq, 0)
        
        t_in = torch.LongTensor(seq).unsqueeze(0).to(self.device)
        t_len = torch.LongTensor([t_in.size(1)]).to(self.device)

        with torch.no_grad():
            # Standard VITS inference call using aux_input for lengths
            outputs = self.model.inference(t_in, aux_input={"x_lengths": t_len})
        
        if isinstance(outputs, dict):
            wav = outputs["model_outputs"][0].data.cpu().numpy()
        else:
            wav = outputs[0].data.cpu().numpy()
            
        wav = wav.flatten()
        wav = wav / (np.max(np.abs(wav)) + 1e-6) 
        wav_16 = (wav * 32767).astype(np.int16)
        wavfile.write(output_file, self.config.audio.sample_rate, wav_16)
        print(f"✅ Success! Generated {output_file}")

if __name__ == "__main__":
    text = input("Enter Dholuo text (or press Enter for default): ").strip()
    if not text:
        text = "japuonj morwa mar bayoloji ne olero kaka ler loso gik moko ka itiyo gi picha ma oting'o weche mathoth"
    
    gender = input("Enter gender (male/female, default: female): ").strip().lower()
    if gender not in ["male", "female"]:
        gender = "female"
    
    dho_tts = DholuoTTS(gender=gender)
    dho_tts.speak(text)