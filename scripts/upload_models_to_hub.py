# scripts/upload_models_to_hub.py
import os
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv

# Load environment variables from .env file
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# --- CONFIGURATION ---
REPO_ID = "sowallah/dholuo-tts-models"
TOKEN = os.getenv("HF_TOKEN")

if not TOKEN:
    raise ValueError("HF_TOKEN not found in .env file. Please create a .env file with: HF_TOKEN=your_token")

api = HfApi()

def upload():
    print(f"🚀 Preparing to upload to repository: {REPO_ID}...")
    print(f"📢 Note: Ensure you have manually created the repository '{REPO_ID}' at huggingface.co/new")



    # 1. Upload POS Model
    pos_dir = os.path.join(PROJECT_ROOT, "models/luo-pos")
    if os.path.exists(pos_dir):
        print(f"📤 Uploading POS model from {pos_dir}...")
        api.upload_folder(
            folder_path=pos_dir,
            repo_id=REPO_ID,
            path_in_repo="luo-pos",
            token=TOKEN
        )
    else:
        print(f"⚠️ POS model not found at {pos_dir}")

    # 2. Upload TTS models
    for gender, checkpoint in [("female", "checkpoint_180000"), ("male", "checkpoint_160000")]:
        gender_dir = os.path.join(PROJECT_ROOT, f"models/luo-tts/{gender}")
        if os.path.exists(gender_dir):
            print(f"📤 Uploading TTS {gender} model (selective)...")
            # Upload config.json
            api.upload_file(
                path_or_fileobj=os.path.join(gender_dir, "config.json"),
                path_in_repo=f"luo-tts/{gender}/config.json",
                repo_id=REPO_ID,
                token=TOKEN
            )
            # Upload checkpoint
            api.upload_file(
                path_or_fileobj=os.path.join(gender_dir, f"{checkpoint}.pth"),
                path_in_repo=f"luo-tts/{gender}/{checkpoint}.pth",
                repo_id=REPO_ID,
                token=TOKEN
            )
        else:
            print(f"⚠️ TTS {gender} model directory not found at {gender_dir}")

    print(f"\n🎉 Essential models uploaded to: https://huggingface.co/{REPO_ID}")

if __name__ == "__main__":
    upload()
