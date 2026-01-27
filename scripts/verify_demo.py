# scripts/verify_demo.py
import sys
import os

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.app import DemoModel

def test_demo():
    print("🧪 Starting Demo Verification...")
    
    try:
        # 1. Initialize
        model = DemoModel()
        model.load()
        
        # 2. Test Synthesis
        text = "Nyithindo ringo e dala"
        print(f"🗣️ Testing synthesis for: '{text}'")
        
        # Run for both genders if available, or just check what's loaded
        genders = list(model.models.keys())
        if not genders:
            print("⚠️ No models loaded! Check paths.")
            return False
            
        for gender in genders:
            print(f"  - Testing {gender}...")
            pos, ipa, audio_path, wave, spec = model.synthesize(text, gender=gender)
            
            if audio_path and os.path.exists(audio_path):
                print(f"  ✅ Audio generated at {audio_path}")
            else:
                print(f"  ❌ Audio generation failed for {gender}")
                return False
                
        print("🎉 Verification Successful! App logic is sound.")
        return True
        
    except Exception as e:
        print(f"❌ detailed error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_demo()
    sys.exit(0 if success else 1)
