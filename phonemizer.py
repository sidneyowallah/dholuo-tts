import re

class Phonemizer:
    def __init__(self, tagger=None):
        """
        Dholuo Grapheme-to-Phoneme (G2P) converter with Tonal Injection.
        
        Args:
            tagger: An instance of the Tagger class. If None, 
                    it can still phonemize individual words without POS awareness.
        """
        self.tagger = tagger
        
        # 1. THE MAPPING
        # Note: 'g' is mapped to 'ɡ' (U+0261) to match the VITS training set exactly.
        # Digraphs are listed to be caught by regex first.
        self.mapping = {
            # Digraphs & Special Consonants
            "ng'": "ŋ",    # ng'ato (velar nasal)
            "ng": "ŋg",    # panga (pre-nasalized velar)
            "ny": "ɲ",     # nyako (palatal nasal)
            "th": "θ",     # thum (voiceless dental plosive/fricative)
            "dh": "ð",     # dho (voiced dental plosive/fricative)
            "ch": "tʃ",    # chiro (voiceless palatal affricate)
            "sh": "ʃ",     # shati (voiceless postalveolar fricative)
            "j": "ɟ",      # voiced palatal plosive (standard Dholuo 'j')
            "y": "j",      # semi-vowel 'y' -> IPA 'j'
            
            # Vowels (Standardized for VITS)
            "a": "a", 
            "e": "ɛ",      
            "i": "i", 
            "o": "ɔ",      
            "u": "u",

            # Standard Consonants
            "b": "b", "p": "p", "m": "m", "w": "w",
            "f": "f", "v": "v", "t": "t", "d": "d",
            "s": "s", "n": "n", "l": "l", "r": "ɾ",
            "k": "k", "g": "g", "h": "h" 
        }

        # Compile a regex to match all keys, longest first (e.g., 'ng'' before 'n')
        pattern = "|".join(re.escape(k) for k in sorted(self.mapping.keys(), key=len, reverse=True))
        self.regex = re.compile(pattern)

    def grapheme_to_phoneme(self, word):
        """Replaces characters based on the mapping in a single pass."""
        return self.regex.sub(lambda m: self.mapping[m.group(0)], word.lower().strip())

    def add_tone(self, phoneme, pos_tag):
        """Injects tone markers based on the POS tag."""
        if pos_tag == "V":
            return phoneme + "˥"  # High tone for Verbs
        elif pos_tag == "NN":
            return phoneme + "˩"  # Low tone for Nouns
        return phoneme

    def phonemize(self, word, pos_tag=None):
        """Processes a single word with optional POS awareness."""
        base_ipa = self.grapheme_to_phoneme(word)
        if pos_tag:
            return self.add_tone(base_ipa, pos_tag)
        return base_ipa

    def phonemize_tagged_pairs(self, tagged_pairs):
        """Converts a list of (word, tag) tuples into a space-separated IPA string."""
        ipa_tokens = []
        for word, tag in tagged_pairs:
            ipa_tokens.append(self.phonemize(word, tag))
        return " ".join(ipa_tokens)

    def phonemize_text(self, text):
        """
        The Full Pipeline:
        1. Tags the raw text (using heuristics + AfroXLMR).
        2. Converts each tagged word to IPA.
        3. Appends tonal pitch based on the tag.
        """
        if self.tagger is None:
            raise ValueError("Phonemizer must be initialized with a Tagger to process raw text.")
            
        tagged_pairs = self.tagger.tag(text)
        return self.phonemize_tagged_pairs(tagged_pairs)

if __name__ == "__main__":
    # Unit Test
    from tagger import Tagger
    
    # Initialize components
    tagger_instance = Tagger()
    phonemizer = Phonemizer(tagger=tagger_instance)
    
    # Test sentence
    test_text = "Nyithindo ringo e dala."
    print(f"Input: {test_text}")
    
    # Get IPA output
    output_ipa = phonemizer.phonemize_text(test_text)
    print(f"Output IPA: {output_ipa}")
    
    # Expected Output Example:
    # ɲiθindɔ˩ ɾiŋɡɔ˥ ɛ dala˩