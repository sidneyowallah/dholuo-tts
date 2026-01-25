import os
import shutil
import pandas as pd
import torch
import librosa
from pydub import AudioSegment
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCTC

# =================CONFIGURATION=================
AUDIO_FOLDER = "data/audio/webm/"
WAV_OUTPUT_FOLDER = "data/audio/wav/"
WAV_MALE_FOLDER = "data/audio/wav_male/"
WAV_FEMALE_FOLDER = "data/audio/wav_female/"
MODEL_ID = "CLEAR-Global/w2v-bert-2.0-luo_cv_fleurs_19h"
TTS_SAMPLE_RATE = 22050  # Best for TTS
ASR_SAMPLE_RATE = 16000  # Required for ASR Model
OUTPUT_CSV = "data/csv/final_dataset.csv"
MALE_CSV = "data/csv/male_dataset.csv"
FEMALE_CSV = "data/csv/female_dataset.csv"
# ===============================================

def split_by_gender(df):
    """Split audio files and metadata by speaker gender"""
    print("\nSplitting dataset by gender...")
    
    os.makedirs(WAV_MALE_FOLDER, exist_ok=True)
    os.makedirs(WAV_FEMALE_FOLDER, exist_ok=True)
    
    # Split dataframes
    df_male = df[df['gender'] == 'male_masculine'].copy()
    df_female = df[df['gender'] == 'female_feminine'].copy()
    
    df_male.to_csv(MALE_CSV, index=False)
    df_female.to_csv(FEMALE_CSV, index=False)
    print(f"CSV split - Male: {len(df_male)} rows, Female: {len(df_female)} rows")
    
    # Split audio files
    gender_map = dict(zip(df['user_id'], df['gender']))
    male_count = female_count = skipped = 0
    
    for filename in tqdm(os.listdir(WAV_OUTPUT_FOLDER), desc="Copying audio"):
        if not filename.endswith('.wav'):
            continue
        
        user_id = filename.rsplit('-', 1)[0]
        gender = gender_map.get(user_id)
        src = os.path.join(WAV_OUTPUT_FOLDER, filename)
        
        if gender == 'male_masculine':
            shutil.copy2(src, os.path.join(WAV_MALE_FOLDER, filename))
            male_count += 1
        elif gender == 'female_feminine':
            shutil.copy2(src, os.path.join(WAV_FEMALE_FOLDER, filename))
            female_count += 1
        else:
            skipped += 1
    
    print(f"Audio split - Male: {male_count}, Female: {female_count}, Skipped: {skipped}")

def main():
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForCTC.from_pretrained(MODEL_ID).to(device)

    if not os.path.exists(WAV_OUTPUT_FOLDER): os.makedirs(WAV_OUTPUT_FOLDER)

    df_audio = pd.read_csv("data/csv/audio.csv")
    df_meta = pd.read_csv("data/csv/users-meta.csv")
    df = pd.merge(df_audio, df_meta.rename(columns={"user-id": "user_id"}), on="user_id", how="left")

    transcriptions = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        orig_name = row['audio_name']
        webm_path = os.path.join(AUDIO_FOLDER, orig_name)
        wav_name = orig_name.replace(".webm", ".wav")
        wav_path = os.path.join(WAV_OUTPUT_FOLDER, wav_name)

        try:
            # 1. Convert to 22.05kHz for TTS Quality
            if os.path.exists(webm_path):
                audio = AudioSegment.from_file(webm_path, format="webm")
                audio.set_frame_rate(TTS_SAMPLE_RATE).set_channels(1).export(wav_path, format="wav")
                
                # 2. Transcribe (Resample to 16k just for the model)
                speech, _ = librosa.load(wav_path, sr=ASR_SAMPLE_RATE)
                inputs = processor(speech, sampling_rate=ASR_SAMPLE_RATE, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inputs).logits
                predicted_ids = torch.argmax(logits, dim=-1)
                transcriptions.append(processor.batch_decode(predicted_ids)[0].lower())
            else:
                transcriptions.append("FILE_NOT_FOUND")
        except Exception as e:
            transcriptions.append(f"ERROR: {e}")

    df['transcription'] = transcriptions
    
    # Cleanup
    cols_to_drop = ['audio_url', 'reviewer_id', 'decision', 'notes', 'time_spent', 'audio_id']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df['audio_name'] = df['audio_name'].str.replace('.webm', '.wav', regex=False)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Transcription complete. Saved to {OUTPUT_CSV}")
    
    # Split by gender
    split_by_gender(df)

if __name__ == "__main__":
    main()