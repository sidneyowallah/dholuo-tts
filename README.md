# Dholuo POS-Aware TTS Pipeline 🇰🇪

![License](https://img.shields.io/badge/license-MIT-green.svg)
![Language](https://img.shields.io/badge/language-Dholuo-blue.svg)
![TTS](https://img.shields.io/badge/TTS-VITS-orange.svg)
![NLP](https://img.shields.io/badge/NLP-AfroXLMR-purple.svg)
![GPU](https://img.shields.io/badge/GPU-RTX%20A5000-76B900.svg)
![Status](https://img.shields.io/badge/status-Training-yellow.svg)

This repository contains a complete pipeline for building a high-fidelity, natural-sounding Text-to-Speech (TTS) system for the **Dholuo** language. Unlike standard multilingual TTS, this system uses a **Part-of-Speech (POS) aware G2P (Grapheme-to-Phoneme)** approach to solve the problem of homograph disambiguation and tonal accuracy.

---

## 🌟 Features

- **Grammatical Disambiguation**
  Uses a fine-tuned **AfroXLMR** model to identify POS tags, allowing the system to distinguish between homographs (e.g., _dhok_ as "mouth" vs. "cows").

- **Tonal Injection**
  Automatically injects tone markers (`˥` for high, `˩` for low) into the phonetic transcription based on grammatical categories (Nouns vs. Verbs).

- **Custom Dholuo G2P**
  A specialized phonemizer that handles Dholuo-specific digraphs (`ny`, `ng'`, `th`, `dh`) and ATR vowel harmony.

- **Modular Architecture**
  Separated concerns with dedicated modules: `phonemizer.py` for G2P, `tagger.py` for POS tagging, and `utils.py` for shared utilities.

- **Comprehensive Testing**
  Full test suite covering unit tests, integration tests, and end-to-end pipeline validation.

- **End-to-End VITS**
  Training on the state-of-the-art **VITS** (Variational Inference with adversarial learning for end-to-end Text-to-Speech) architecture.

---

## 🏗️ Architecture

```mermaid
graph TD
    subgraph "1. Data Collection & Transcription"
        A[DhoNam Raw Audio .webm] --> B[Audio Conversion: 22.05kHz .wav]
        B --> C[ASR: W2V-BERT 2.0 Transcription]
    end

    subgraph "2. NLP & Grammatical Enrichment"
        D[KenPOS Dataset stored as dh.parquet] --> E[AfroXLMR-76L Fine-tuning]
        C & E --> F[POS Tagging: Word_TAG format]
    end

    subgraph "3. Phonology & G2P Logic"
        F --> G[Custom Dholuo Regex G2P]
        G --> H[Tone Injector: Noun/Low vs. Verb/High]
        H --> I[POS-Aware IPA Lexicon]
    end

    subgraph "4. Neural Acoustic Modeling (VITS)"
        I --> J[Stochastic Duration Predictor]
        J --> K[Flow-based Decoder]
        K --> L[HiFi-GAN Discriminator]
        L --> M[Acoustic Feature Alignment]
    end

    M --> N((Final Dholuo Speech))

    style N fill:#f96,stroke:#333,stroke-width:4px
```

### NLP Phase

- Fine-tune `Davlan/afro-xlmr-large-76L` on the **KenPOS Dholuo** dataset.

### Preprocessing Phase

- Transcribe **DhoNam** audio using **W2V-BERT 2.0**.
- Filter for high-confidence recordings (>80%).
- Tag transcripts with the POS model (`Word_TAG` format).

### G2P Phase

- Convert `Word_TAG` units into **IPA phonemes** with associated pitch markers.

### Acoustic Phase

- Train **VITS** on RunPod (NVIDIA RTX A5000) to map **IPA + Tones** to raw audio.

---

## 📁 Project Structure

```
dholuo_tts/
├── phonemizer.py          # G2P converter with tone injection
├── tagger.py              # POS tagging with AfroXLMR
├── utils.py               # Shared utilities and custom VITS model
├── generate_lexicon.py    # Bulk IPA lexicon generator
├── preprocess_dhonam.py   # Audio preprocessing pipeline
├── train_vits.py          # VITS training script
├── tests/                 # Comprehensive test suite
│   ├── test_phonemizer.py
│   ├── test_integration.py
│   ├── test_phonemizer_with_tagger.py
│   ├── test_model.py      # End-to-end TTS inference test
│   └── output/            # Generated test audio files
├── data/
│   ├── dholuo_lexicon.json  # POS-aware IPA dictionary
│   └── csv/                 # Training metadata
└── models/
    ├── luo-pos/             # Fine-tuned POS tagger
    └── luo-tts/             # VITS checkpoints
```

---

## 🚀 Getting Started

### 🔧 Installation & Setup

#### 1. System Dependencies

The TTS engine requires **espeak-ng** for phoneme processing:

```bash
sudo apt-get update && sudo apt-get install espeak-ng -y
```

#### 2. Python Environment

Install required Python packages using `uv`:

```bash
uv sync
```

Or with pip:

```bash
pip install TTS pandas transformers accelerate torchaudio tqdm
```

---

## 🛠️ Pipeline Execution

### Phase 1: POS Tagger Fine-Tuning

Fine-tune the `Davlan/afro-xlmr-large-76L` model using the **KenPOS** dataset (`dh.parquet`).

```bash
uv run train_pos_local.py   # For local training
uv run train_pos_cloud.py   # For cloud training
```

### Phase 2: Metadata Preprocessing

Transcribe the DhoNam audio files and apply POS tags to create the training metadata.

```bash
uv run transcribe_audio.py
uv run preprocess_dhonam.py
```

### Phase 3: Lexicon Generation

Generate the IPA phonetic dictionary with POS-aware tone markers.

```bash
uv run generate_lexicon.py
```

This creates `data/dholuo_lexicon.json` with entries like:

```json
{
  "nam_NN": "nam˩",
  "ringo_V": "ɾiŋgɔ˥",
  "dho_NN": "ðɔ˩"
}
```

### Phase 4: VITS Metadata Creation

Convert POS-tagged metadata to IPA phonemes for VITS training.

```bash
uv run create_vits_metadata.py
```

### Phase 5: VITS Training

Train the acoustic model on an NVIDIA GPU.

```bash
python train_vits.py
```

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Test phonemizer G2P and tone injection
uv run python tests/test_phonemizer.py

# Test integration with tagged pairs
uv run python tests/test_integration.py

# Test end-to-end with mock tagger
uv run python tests/test_phonemizer_with_tagger.py

# Test full TTS model inference
uv run python tests/test_model.py
```

---

## 🔧 Module Usage

### Phonemizer

```python
from phonemizer import Phonemizer

# Initialize without tagger for direct phonemization
p = Phonemizer(tagger=False)

# Convert word with POS tag to IPA
ipa = p.phonemize("dho", "NN")  # Returns: "ðɔ˩"

# Process tagged pairs
tagged = [("nyithindo", "NN"), ("ringo", "V")]
result = p.phonemize_tagged_pairs(tagged)  # Returns: "ɲiθindɔ˩ ɾiŋgɔ˥"
```

### Tagger

```python
from tagger import Tagger

# Initialize POS tagger
tagger = Tagger()

# Tag raw text
tagged_pairs = tagger.tag("Nyithindo ringo e dala")
# Returns: [("nyithindo", "NN"), ("ringo", "V"), ("e", "P"), ("dala", "NN")]
```

### End-to-End Pipeline

```python
from phonemizer import Phonemizer
from tagger import Tagger

# Initialize components
tagger = Tagger()
phoneme = Phonemizer(tagger=tagger)

# Process raw text to IPA
text = "Nyithindo ringo e dala"
ipa_output = phoneme.phonemize_text(text)
print(ipa_output)  # "ɲiθindɔ˩ ɾiŋgɔ˥ ɛ dala˩"
```

---

## 📂 Dataset Credits

- **KenPOS**: Dholuo Part-of-Speech dataset (locally stored as `dh.parquet`).
- **DhoNam**: Dholuo Speech dataset used for ASR and TTS training.
- **AfroXLMR**: Pre-trained multilingual model by Davlan, used as the backbone for Nilotic NLP tasks.

---

## ⚖️ License

This project is licensed under the **MIT License** — see the `LICENSE` file for details.

---

**Author:** Sidney Owallah
**Collaborators:** Trained on RunPod RTX A5000
