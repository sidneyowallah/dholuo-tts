import string
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from preprocess_dhonam import DHOLUO_STOPWORDS

class Tagger:
    def __init__(self, model_path="models/luo-pos", base_model="Davlan/afro-xlmr-large-76L"):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForTokenClassification.from_pretrained(model_path)
        
        self.tagger_pipe = pipeline(
            "token-classification", 
            model=model, 
            tokenizer=tokenizer,
            aggregation_strategy="simple", 
            device=self.device
        )
        self.verb_prefixes = ('a', 'i', 'o', 'wa', 'u', 'gi', 'ko')

    def tag(self, text):
        """Matches the logic from preprocess_dhonam.py exactly"""
        raw_words = text.replace("’", "'").replace("‘", "'").split()
        results = self.tagger_pipe(text)
        tagged_pairs = []

        for word in raw_words:
            clean_word = word.strip().lower().translate(
                str.maketrans('', '', string.punctuation.replace("'", ""))
            )
            match_word = clean_word.replace("'", "")
            if not match_word: continue

            tag = "UNK"
            for res in results:
                model_token = res['word'].replace(" ", "").replace("'", "").lower()
                if model_token in match_word or match_word in model_token:
                    tag = res['entity_group']
                    break
            
            # --- CUSTOM FALLBACK LOGIC ---
            if tag == "UNK":
                if clean_word in DHOLUO_STOPWORDS or match_word in DHOLUO_STOPWORDS:
                    tag = DHOLUO_STOPWORDS.get(clean_word, DHOLUO_STOPWORDS.get(match_word))
                elif clean_word.endswith("ruok"): tag = "NN"
                elif clean_word.startswith("ko") and len(clean_word) > 4: tag = "V"
                elif clean_word.startswith("jo"): tag = "NN"
                elif clean_word.startswith("ma") and len(clean_word) > 3: tag = "ADJ"
                elif clean_word.startswith(self.verb_prefixes) and (clean_word.endswith("o") or len(clean_word) > 4):
                    tag = "V"
                elif len(clean_word) <= 2: tag = "PRON"
                else: tag = "NN" # Safety default
            
            tagged_pairs.append((clean_word, tag))
        return tagged_pairs