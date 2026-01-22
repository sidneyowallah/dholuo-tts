import os
import sys
import torch
import torchaudio
import soundfile as sf
import json

# ======================================================
# 1. AUDIO PATCH (Bypass torchcodec)
# ======================================================
def patched_load_audio(file_path):
    data, sr = sf.read(file_path, dtype='float32')
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.transpose(0, 1)
    return tensor, sr

sys.modules['torchcodec'] = None
sys.modules['torchcodec.decoders'] = None
import TTS.tts.models.vits as vits_module
vits_module.load_audio = patched_load_audio
torchaudio.load = patched_load_audio

# ======================================================
# 2. IMPORTS
# ======================================================
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import CharactersConfig, BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.text import cleaners
from TTS.utils.audio import AudioProcessor # <--- NEW IMPORT

# ======================================================
# 3. UTILS & CUSTOM CLASS
# ======================================================
class AttributeDict(dict):
    def __getattr__(self, name):
        try: return self[name]
        except KeyError: raise AttributeError(f"No attribute '{name}'")

class DholuoVits(Vits):
    def __init__(self, config):
        super().__init__(config)
        # Force tokenizer initialization
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            config.phoneme_language = "en"
            res = TTSTokenizer.init_from_config(config)
            self.tokenizer = res[0] if isinstance(res, tuple) else res
            config.phoneme_language = None
        self.tokenizer.text_cleaner = getattr(cleaners, "basic_cleaners")
        self.tokenizer.use_phonemes = False

# ======================================================
# 4. PATHS & CONFIG
# ======================================================
dataset_path = "/workspace"
metadata_file = os.path.join(dataset_path, "vits_train_meta.csv")

audio_params = {
    "sample_rate": 22050, "win_length": 1024, "hop_length": 256,
    "fft_size": 1024, "num_mels": 80, "resample": True,
    "mel_fmin": 0, "mel_fmax": None,
}
audio_config = AttributeDict(audio_params)

config = VitsConfig(
    audio=audio_config,
    run_name="vits_dholuo_pos_aware",
    batch_size=16,
    eval_batch_size=8,
    num_loader_workers=0,
    num_eval_loader_workers=0,
    run_eval=True,
    epochs=1000,
    text_cleaner="basic_cleaners", 
    use_phonemes=False,
    characters=CharactersConfig(
        pad="_", eos="~", bos="^",
        characters="abcdefghijklmnopqrstuvwxyzɛɔɟɲŋθðɾtʃdʒ˩˥",
        punctuations=".,!?- ",
    ),
    output_path="/workspace/training_output/",
)

# ======================================================
# 5. DATASET LOADING
# ======================================================
def custom_formatter(root_path, meta_file, **kwargs):
    items = []
    with open(meta_file, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.strip().split("|")
            if len(cols) >= 3:
                items.append({
                    "text": cols[2], 
                    "audio_file": os.path.join(root_path, "wavs", cols[0] + ".wav"),
                    "root_path": root_path, "speaker_name": "dhonam", "language": "en"
                })
    return items

dataset_config = BaseDatasetConfig(
    dataset_name="dholuo_dhonam", meta_file_train=metadata_file, path=dataset_path,
)

train_samples, eval_samples = load_tts_samples(
    dataset_config, eval_split=True, eval_split_max_size=100,
    eval_split_size=0.02, formatter=custom_formatter
)

# ======================================================
# 6. INITIALIZE MODEL & ATTACH MATH ENGINE (THE FIX)
# ======================================================
print("🔧 Initializing Model and Audio Processor...")
model = DholuoVits(config)

# THE FIX: Manually create the AudioProcessor and attach it to the model
# This allows the model to calculate spectrograms for logging
ap = AudioProcessor(**audio_params)
model.ap = ap 

# ======================================================
# 7. TRAINER
# ======================================================
trainer_args = TrainerArgs()
# Dynamically set checkpoint limits to save space
for attr in ["save_step", "checkpoint_step"]:
    if hasattr(trainer_args, attr): setattr(trainer_args, attr, 5000)
for attr in ["keep_checkpoint_max", "save_n_checkpoints"]:
    if hasattr(trainer_args, attr): setattr(trainer_args, attr, 2)

trainer = Trainer(
    trainer_args, config, output_path="/workspace/training_output/",
    model=model, train_samples=train_samples, eval_samples=eval_samples,
)

print("\n🚀 STARTING DHOLUO POS-AWARE TRAINING LOOP...")
trainer.fit()