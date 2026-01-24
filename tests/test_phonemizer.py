import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonemizer import Phonemizer

def test_phonemizer():
    p = Phonemizer(tagger=False)  # Disable auto-tagger for unit tests
    
    # Test 1: Basic G2P
    assert p.grapheme_to_phoneme("dho") == "ðɔ", "Failed: dho -> ðɔ"
    assert p.grapheme_to_phoneme("nam") == "nam", "Failed: nam -> nam"
    assert p.grapheme_to_phoneme("ng'a") == "ŋa", "Failed: ng'a -> ŋa"
    assert p.grapheme_to_phoneme("nyako") == "ɲakɔ", "Failed: nyako -> ɲakɔ"
    
    # Test 2: Tone injection
    assert p.add_tone("nam", "NN") == "nam˩", "Failed: Noun tone"
    assert p.add_tone("ringo", "V") == "ringo˥", "Failed: Verb tone"
    assert p.add_tone("e", "P") == "e", "Failed: No tone for preposition"
    
    # Test 3: Full phonemization
    assert p.phonemize("dho", "NN") == "ðɔ˩", "Failed: dho_NN"
    assert p.phonemize("ringo", "V") == "ɾiŋgɔ˥", "Failed: ringo_V"
    
    # Test 4: Tagged pairs
    tagged = [("nyithindo", "NN"), ("ringo", "V"), ("e", "P"), ("dala", "NN")]
    result = p.phonemize_tagged_pairs(tagged)
    expected = "ɲiθindɔ˩ ɾiŋgɔ˥ ɛ dala˩"
    assert result == expected, f"Failed: {result} != {expected}"
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_phonemizer()
