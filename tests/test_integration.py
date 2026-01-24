import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonemizer import Phonemizer

def test_integration():
    """Test that phonemizer works as expected in the pipeline"""
    p = Phonemizer(tagger=False)  # Disable auto-tagger for unit tests
    
    # Simulate tagged pairs from POS tagger
    tagged_pairs = [
        ("nyithindo", "NN"),  # children (noun)
        ("ringo", "V"),        # run (verb)
        ("e", "P"),            # in (preposition)
        ("dala", "NN")         # home (noun)
    ]
    
    # Test individual phonemization
    print("Individual word phonemization:")
    for word, tag in tagged_pairs:
        phoneme = p.phonemize(word, tag)
        print(f"  {word}_{tag} -> {phoneme}")
    
    # Test full sentence phonemization
    sentence_ipa = p.phonemize_tagged_pairs(tagged_pairs)
    print(f"\nFull sentence: {sentence_ipa}")
    
    # Verify expected output
    expected = "ɲiθindɔ˩ ɾiŋgɔ˥ ɛ dala˩"
    assert sentence_ipa == expected, f"Mismatch: {sentence_ipa} != {expected}"
    
    print("\n✅ Integration test passed!")

if __name__ == "__main__":
    test_integration()
