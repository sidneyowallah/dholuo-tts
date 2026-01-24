import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phonemizer import Phonemizer

def test_with_mock_tagger():
    """Test phonemizer with a mock tagger"""
    
    class MockTagger:
        def tag(self, text):
            # Mock tagging for "Nyithindo ringo e dala"
            return [("nyithindo", "NN"), ("ringo", "V"), ("e", "P"), ("dala", "NN")]
    
    # Test without tagger
    p = Phonemizer(tagger=False)
    result = p.phonemize("dho", "NN")
    assert result == "ðɔ˩", f"Failed: {result}"
    print("✅ Phonemizer works without tagger")
    
    # Test with tagger
    p_with_tagger = Phonemizer(tagger=MockTagger())
    result = p_with_tagger.phonemize_text("Nyithindo ringo e dala")
    expected = "ɲiθindɔ˩ ɾiŋgɔ˥ ɛ dala˩"
    assert result == expected, f"Failed: {result} != {expected}"
    print("✅ End-to-end phonemization works!")
    print(f"   Input:  'Nyithindo ringo e dala'")
    print(f"   Output: '{result}'")

if __name__ == "__main__":
    test_with_mock_tagger()
