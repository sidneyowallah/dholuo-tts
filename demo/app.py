# demo/app.py
import os
import sys
import time
import json
import torch
import numpy as np
import gradio as gr
from scipy.io import wavfile

# Add parent directory to path to import root modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tagger import Tagger
from phonemizer import Phonemizer
from demo.components import Header, InputSection, OptionsPanel, OutputSection, InfoPanel
from demo.visualizations import create_pos_html, plot_waveform, plot_spectrogram

# --- MODEL IMPORTS (Copied/Adapted from tests/test_model.py to avoid dependency on tests folder) ---
from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.text import cleaners
from TTS.utils.audio import AudioProcessor

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINTS = {
    "female": "checkpoint_180000",
    "male": "checkpoint_160000"
}

# --- MODEL CLASSES ---
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

class DemoModel:
    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.models = {}
        self.configs = {}
        self.ap = None
        self.tagger = None
        self.phonemizer = None
        self.loaded = False

    def load(self):
        if self.loaded: return
        print("🚀 Loading Models...")
        
        # Load Resources
        self.tagger = Tagger(model_path=os.path.join(PROJECT_ROOT, "models/luo-pos"))
        self.phonemizer = Phonemizer(tagger=self.tagger)
        
        HUB_REPO = "sowallah/dholuo-tts-models"
        from huggingface_hub import hf_hub_download

        # Load Male and Female Models
        for gender in ["female", "male"]:
            config_path = os.path.join(PROJECT_ROOT, f"models/luo-tts/{gender}/config.json")
            checkpoint_name = CHECKPOINTS.get(gender, "checkpoint_180000")
            model_path = os.path.join(PROJECT_ROOT, f"models/luo-tts/{gender}/{checkpoint_name}.pth")
            
            # Hub check
            if not os.path.exists(config_path):
                print(f"📥 Downloading {gender} config from Hub...")
                try:
                    config_path = hf_hub_download(repo_id=HUB_REPO, filename=f"luo-tts/{gender}/config.json")
                    model_path = hf_hub_download(repo_id=HUB_REPO, filename=f"luo-tts/{gender}/{checkpoint_name}.pth")
                except Exception as e:
                    print(f"⚠️ Warning: Model for {gender} not found on Hub: {e}")
                    continue

            # Config
            train_characters = "abcdefghijklmnopqrstuvwxyzɛɔɟɲŋθðɾtʃdʒ˩˥"
            config = VitsConfig(
                audio=None, run_eval=True, text_cleaner="basic_cleaners",
                use_phonemes=False,
                characters=CharactersConfig(
                    pad="_", eos="~", bos="^", characters=train_characters, punctuations=".,!?- ",
                )
            )
            config.load_json(config_path)
            
            # Init Audio Processor (Shared)
            if self.ap is None:
                self.ap = AudioProcessor(**config.audio)
            
            # Init Model
            model = DholuoVits(config)
            checkpoint = torch.load(model_path, map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            model.to(self.device).eval()
            
            self.models[gender.lower()] = model
            self.configs[gender.lower()] = config
            
        self.loaded = True
        print("✅ Models Loaded!")

    def synthesize(self, text, gender="female", speed=1.0, noise_scale=0.667):
        if not text.strip():
            return None, "", None, None, None

        gender = gender.lower()
        model = self.models.get(gender)
        config = self.configs.get(gender)
        
        if not model:
            return None, f"Error: {gender} model not loaded.", None, None, None
            
        # Pipeline Steps
        # 1. Tagging & Phonemization
        start_time = time.time()
        tagged_pairs = self.tagger.tag(text)
        ipa = self.phonemizer.phonemize_text(text)
        
        # 2. Synthesis
        seq = model.tokenizer.encode(ipa)
        if config.add_blank:
            seq = intersperse(seq, 0)
            
        t_in = torch.LongTensor(seq).unsqueeze(0).to(self.device)
        t_len = torch.LongTensor([t_in.size(1)]).to(self.device)
        
        with torch.no_grad():
            outputs = model.inference(
                t_in, 
                aux_input={"x_lengths": t_len}
                # noise_scale=noise_scale, # Not supported in this version
                # length_scale=1.0/speed   # Not supported in this version
            )
            
        wav = outputs["model_outputs"][0].data.cpu().numpy().flatten() if isinstance(outputs, dict) else outputs[0].data.cpu().numpy().flatten()
        wav = wav / (np.max(np.abs(wav)) + 1e-6)
        
        # File generation
        assets_dir = os.path.join(PROJECT_ROOT, "demo/assets")
        os.makedirs(assets_dir, exist_ok=True)
        output_file = os.path.join(assets_dir, "output.wav")
        
        wav_16 = (wav * 32767).astype(np.int16)
        wavfile.write(output_file, config.audio.sample_rate, wav_16)
        
        print(f"⏱️ Synthesis time: {time.time() - start_time:.2f}s")
        
        # Visuals
        pos_html = create_pos_html(text, tagged_pairs)
        waveform = plot_waveform(output_file)
        spectrogram = plot_spectrogram(output_file)
        
        return pos_html, f"**/{ipa}/**", output_file, waveform, spectrogram

# --- APP INSTANCE ---
demo_model = DemoModel()

# Read CSS from file
css_path = os.path.join(PROJECT_ROOT, "demo/styles.css")
with open(css_path, "r") as f:
    custom_css = f.read()

with gr.Blocks(title="Dholuo TTS") as app:
    Header()
    
    with gr.Row():
        with gr.Column(scale=4):
            text_input = InputSection()
        with gr.Column(scale=2):
            voice, speed, noise = OptionsPanel()
            
    generate_btn = gr.Button("🎤 Generate Speech", variant="primary", scale=1)
    
    pos_view, ipa_view, audio_player, wave_view, spec_view = OutputSection()
    
    InfoPanel()
    
    # Event Handler
    def process_request(text, voice_sel, speed_val, noise_val):
        if not demo_model.loaded:
            demo_model.load()
        return demo_model.synthesize(text, voice_sel, speed_val, noise_val)

    generate_btn.click(
        fn=process_request,
        inputs=[text_input, voice, speed, noise],
        outputs=[pos_view, ipa_view, audio_player, wave_view, spec_view]
    )

if __name__ == "__main__":
    demo_model.load()
    app.launch(server_name="0.0.0.0", server_port=7860, css=custom_css, theme=gr.themes.Soft())
