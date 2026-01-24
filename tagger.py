import string
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Dholuo stopwords dictionary
DHOLUO_STOPWORDS = {
    # --- PRONOUNS (PRON) ---
    "an": "PRON", "emane": "PRON", "en": "PRON", "endo": "PRON", "gin": "PRON", 
    "in": "PRON", "ingi": "PRON", "mekgi": "PRON", "ne": "PRON", "nga": "PRON", 
    "ngama": "PRON", "ngata": "PRON", "un": "PRON", "wan": "PRON",
    # --- NOUNS (NN) ---
    "arus": "NN", "barua": "NN", "betri": "NN", "bote": "NN", "chamcham": "NN", 
    "chieng": "NN", "chieng'wa": "NN", "dala": "NN", "dend": "NN", "dhano": "NN", 
    "doo": "NN", "duond": "NN", "duong'": "NN", "dwe": "NN", "gima": "NN", 
    "gokinyi": "NN", "gweng": "NN", "higa": "NN", "homwake": "NN", "jangad": "NN", 
    "ji": "NN", "kauntiy": "NN", "kongo": "NN", "kuguok": "NN", "lee": "NN", 
    "lesonde": "NN", "lo": "NN", "lowo": "NN", "luwor": "NN", "meru": "NN", 
    "mutasi": "NN", "nam": "NN", "nairobi": "NN", "ndalo": "NN", "ngad": "NN", 
    "ngano": "NN", "ngat": "NN", "ng'at": "NN", "ngato": "NN", "ng'ato": "NN", 
    "ng'atno": "NN", "nge": "NN", "ngere": "NN", "nguon": "NN", "nguono": "NN", 
    "ngut": "NN", "ngwech": "NN", "nyak": "NN", "nyasaye": "NN", "nyathi": "NN", 
    "nying": "NN", "nyithindo": "NN", "nyoro": "NN", "okek": "NN", "ot": "NN", 
    "ouma": "NN", "piny": "NN", "protin": "NN", "rangi": "NN", "ranginy": "NN", 
    "richo": "NN", "sikul": "NN", "taa": "NN", "towel": "NN", "tudruok": "NN", 
    "wenyiew": "NN", "wiya": "NN", "yat": "NN", "chambwa": "NN",
    # --- VERBS (V) ---
    "bedoe": "V", "biro": "V", "chakre": "V", "cham": "V", "chamo": "V", 
    "chango": "V", "chano": "V", "chiemo": "V", "chiwo": "V", "chulo": "V", 
    "chuoyo": "V", "dhawo": "V", "dhiye": "V", "dino": "V", "dinuoya": "V", 
    "dongo": "V", "duoko": "V", "dwaro": "V", "gaye": "V", "gengo": "V", 
    "godoye": "V", "hero": "V", "hinya": "V", "kayawo": "V", "kendo": "V", 
    "kwayo": "V", "lam": "V", "lamo": "V", "limo": "V", "loso": "V", 
    "luok": "V", "luoro": "V", "luongo": "V", "luwo": "V", "miyo": "V", 
    "mobedo": "V", "modong'": "V", "mogik": "V", "moloyo": "V", "mondiki": "V", 
    "motegni": "V", "nang'iew": "V", "ndiko": "V", "neno": "V", "ngado": "V", 
    "nganyo": "V", "nganjo": "V", "ngieyo": "V", "ngiewo": "V", "nginjo": "V", 
    "ngiyo": "V", "ngol": "V", "ngolo": "V", "ngony": "V", "ngeyo": "V", 
    "ningogi": "V", "nitie": "V", "noduokie": "V", "nondiko": "V", "nwa": "V", 
    "nyalo": "V", "nyiewo": "V", "nyisi": "V", "parore": "V", "pidho": "V", 
    "pog": "V", "pongo": "V", "riembe": "V", "ringo": "V", "riwo": "V", 
    "romo": "V", "romre": "V", "tayowa": "V", "tedo": "V", "tim": "V", 
    "timo": "V", "timre": "V", "tingo": "V", "tiyogo": "V", "tony": "V", 
    "tugo": "V", "wuok": "V", "wuoth": "V", "yie": "V", "yudo": "V", 
    "yudore": "V",
    # --- ADJECTIVES (ADJ) ---
    "mongere": "ADJ", "ngeny": "ADJ", "ngich": "ADJ",
    # --- ADVERBS (ADV) ---
    "bange": "ADV", "bangeyo": "ADV", "chon": "ADV", "ka": "ADV", 
    "kane": "ADV", "kanyiyo": "ADV", "kapok": "ADV", "kariembo": "ADV", 
    "kinde": "ADV", "machiegni": "ADV", "machon": "ADV", "malo": "ADV", 
    "nene": "ADV", "ngar": "ADV", "nyime": "ADV",
    # --- DETERMINERS (DET) ---
    "duto": "DET", "ma": "DET", "mane": "DET", "moko": "DET", "yodno": "DET",
    # --- ADPOSITIONS (ADP) ---
    "e": "ADP", "gi": "ADP", "kaw": "ADP", "kod": "ADP", "kuom": "ADP", 
    "nyaka": "ADP",
    # --- CONJUNCTIONS (CONJ) ---
    "koso": "CONJ", "to": "CONJ",
    # --- PARTICLES (PART) ---
    "be": "PART", "bende": "PART", "ni": "PART", "ok": "PART"
}

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