# demo/visualizations.py
import base64
import io
import librosa
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# POS Color Map (Matching Styles)
POS_colors = {
    "NN": "#4285F4",    # Blue
    "V": "#EA4335",     # Red
    "ADJ": "#34A853",   # Green
    "ADV": "#FBBC05",   # Yellow/Orange
    "PRON": "#9C27B0",  # Purple
    "ADP": "#009688",   # Teal
    "DET": "#607D8B",   # Grey-Blue
    "CONJ": "#E91E63",  # Pink
    "PART": "#FF5722",  # Deep Orange
    "UNK": "#9E9E9E"    # Grey
}

def create_pos_html(text, tagged_pairs):
    """
    Generates HTML for colored POS tags.
    """
    html_parts = ['<div class="pos-container">']
    
    for word, tag in tagged_pairs:
        color = POS_colors.get(tag, POS_colors["UNK"])
        # Using the classes defined in styles.css
        html_parts.append(
            f'<div class="token tag-{tag}" title="{tag}">'
            f'<div class="token-text">{word}</div>'
            f'<div class="token-tag">{tag}</div>'
            f'</div>'
        )
    
    html_parts.append('</div>')
    return "".join(html_parts)

def plot_waveform(audio_path):
    """
    Generates a waveform plot using Matplotlib.
    Returns a PIL Image or Plotly Figure.
    """
    y, sr = librosa.load(audio_path, sr=None)
    fig = plt.figure(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, color="#4285F4", alpha=0.8)
    plt.axis("off")
    plt.tight_layout()
    
    return fig

def plot_spectrogram(audio_path):
    """
    Generates a mel-spectrogram plot.
    """
    y, sr = librosa.load(audio_path, sr=None)
    # Mel Spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(10, 3))
    librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, cmap='magma')
    plt.axis("off")
    plt.tight_layout()
    
    return fig

def plot_interactive_spectrogram(audio_path):
    """
    Generates an interactive Plotly spectrogram (Optional, heavier).
    """
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    fig = go.Figure(data=go.Heatmap(
        z=S_dB,
        colorscale='Magma'
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_visible=False,
        yaxis_visible=False,
        height=300
    )
    return fig
