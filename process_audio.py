import os
import pandas as pd
import torch
import librosa
from pydub import AudioSegment
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCTC

# =================CONFIGURATION=================
AUDIO_FOLDER = "data/audio/webm/"           # Folder where .webm files from DhoNam: Dholuo Speech dataset are stored
WAV_OUTPUT_FOLDER = "data/audio/wav/"  # Folder to save converted .wav files

# --- MODEL SETTINGS ---
# The Open Source Dholuo Model
MODEL_ID = "CLEAR-Global/w2v-bert-2.0-luo_cv_fleurs_19h"

# --- TESTING SETTINGS ---
TEST_MODE = False          # Set to True to test, False to run full dataset
TEST_LIMIT = 5            # How many files to process in test mode
# ------------------------

if TEST_MODE:
    OUTPUT_CSV = "data/csv/test_dataset.csv"
else:
    OUTPUT_CSV = "data/csv/final_dataset.csv"
# ===============================================

def main():
    # 1. SETUP HARDWARE & LOAD MODEL
    print("Setting up hardware and loading AI model...")
    
    # Check for Apple Silicon (MPS) or NVIDIA (CUDA) to make this fast
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ Using Apple M1/M2/M3 Acceleration (MPS)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("✅ Using NVIDIA Acceleration (CUDA)")
    else:
        device = "cpu"
        print("⚠️  Using CPU (This might be slow)")

    try:
        # Load the Processor (handles audio formatting) and Model (handles math)
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForCTC.from_pretrained(MODEL_ID).to(device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Create output folder for WAVs if it doesn't exist
    if not os.path.exists(WAV_OUTPUT_FOLDER):
        os.makedirs(WAV_OUTPUT_FOLDER)

    # Load CSVs
    try:
        df_meta = pd.read_csv("data/csv/users-meta.csv")
        df_audio = pd.read_csv("data/csv/audio.csv")
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV files. {e}")
        return

    # 2. MERGE DATAFRAMES
    print("Merging CSV files...")
    if 'user-id' in df_meta.columns:
        df_meta = df_meta.rename(columns={"user-id": "user_id"})
    
    combined_df = pd.merge(df_audio, df_meta, on="user_id", how="left")

    # --- APPLY TEST LIMIT IF ENABLED ---
    if TEST_MODE:
        print(f"\n⚠️  TEST MODE ACTIVE: Processing only first {TEST_LIMIT} rows.")
        combined_df = combined_df.head(TEST_LIMIT)
    else:
        print(f"Processing full dataset: {len(combined_df)} files.")
    # -----------------------------------

    # 3. PROCESSING LOOP (Convert -> Transcribe Locally)
    transcriptions = []
    
    print(f"Starting local transcription using {MODEL_ID}...")
    
    for index, row in tqdm(combined_df.iterrows(), total=combined_df.shape[0]):
        
        original_filename = row['audio_name']
        
        # Paths
        webm_path = os.path.join(AUDIO_FOLDER, original_filename)
        wav_filename = original_filename.replace(".webm", ".wav")
        wav_path = os.path.join(WAV_OUTPUT_FOLDER, wav_filename)

        try:
            # Step A: Convert WEBM to WAV (Standardize format)
            if not os.path.exists(wav_path):
                if os.path.exists(webm_path):
                    audio = AudioSegment.from_file(webm_path, format="webm")
                    # Export as 16kHz Mono (Required for most AI models)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    audio.export(wav_path, format="wav")
                else:
                    transcriptions.append("FILE_NOT_FOUND")
                    continue
            
            # Step B: Local Inference (The Replacement for ElevenLabs)
            if os.path.exists(wav_path):
                # 1. Load audio with librosa (ensures 16000Hz sample rate)
                audio_input, sample_rate = librosa.load(wav_path, sr=16000)
                
                # 2. Process inputs
                inputs = processor(
                    audio_input, 
                    sampling_rate=16000, 
                    return_tensors="pt"
                ).to(device) # Move data to GPU
                
                # 3. Predict (No Gradients needed for inference)
                with torch.no_grad():
                    logits = model(**inputs).logits
                
                # 4. Decode (Convert math IDs to Text)
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription_text = processor.batch_decode(predicted_ids)[0]
                
                transcriptions.append(transcription_text)
            else:
                transcriptions.append("CONVERSION_FAILED")

        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            print(error_msg)
            transcriptions.append(error_msg)

    # 4. SAVE RESULTS
    combined_df['transcription'] = transcriptions
    
    # --- CLEANUP ---
    cols_to_drop = ['audio_url', 'reviewer_id', 'decision', 'notes', 'time_spent']
    print(f"Dropping columns: {cols_to_drop}...")
    combined_df = combined_df.drop(columns=cols_to_drop, errors='ignore')

    print("Updating audio filenames in CSV...")
    if 'audio_name' in combined_df.columns:
        combined_df['audio_name'] = combined_df['audio_name'].str.replace('.webm', '.wav', regex=False)

    combined_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nDone! Processed data saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()