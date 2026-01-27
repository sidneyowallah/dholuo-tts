# demo/components.py
import gradio as gr
from .examples import EXAMPLES, EXAMPLE_MAP

def Header():
    return gr.Markdown(
        """
        # 🇰🇪 Dholuo TTS Demo
        *Context-Aware Text-to-Speech synthesis for the Dholuo language.*
        """
    )

def InputSection():
    with gr.Column():
        text_input = gr.Textbox(
            label="Input Text",
            placeholder="Ketho nwang'o yom, gerru e matek... (Enter Dholuo text here)",
            lines=3,
            elem_classes=["gr-text-input"]
        )
        
        with gr.Row():
            example_dropdown = gr.Dropdown(
                choices=[ex[0] for ex in EXAMPLES],
                label="Load Example",
                value=lambda: None
            )
            clear_btn = gr.Button("Clear", variant="secondary")
            
    # Event wiring
    example_dropdown.change(
        fn=lambda x: x,
        inputs=[example_dropdown],
        outputs=[text_input]
    )
    clear_btn.click(
        fn=lambda: ("", None),
        outputs=[text_input, example_dropdown]
    )
    
    return text_input

def OptionsPanel():
    with gr.Accordion("⚙️ Advanced Options", open=False):
        with gr.Row():
            voice_select = gr.Radio(
                choices=["Female", "Male"],
                value="Female",
                label="Voice Gender"
            )
            speed_slider = gr.Slider(
                minimum=0.5,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="Speech Rate"
            )
            noise_scale = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.667,
                step=0.1,
                label="Noise Scale (Variance)"
            )
            
    return voice_select, speed_slider, noise_scale

def OutputSection():
    with gr.Row():
        with gr.Column(scale=1):
            pos_output = gr.HTML(label="POS Tags")
            ipa_output = gr.Markdown(label="IPA Transcription")
            
        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Synthesized Speech",
                type="filepath",
                interactive=False,
                autoplay=True
            )
            
    with gr.Row():
        waveform_plot = gr.Plot(label="Waveform")
        spectrogram_plot = gr.Plot(label="Spectrogram")
        
    return pos_output, ipa_output, audio_output, waveform_plot, spectrogram_plot

def InfoPanel():
    with gr.Accordion("ℹ️ How it works", open=False):
        gr.Markdown(
            """
            ### Pipeline Process
            1. **Text Normalization**: Cleans and standardizes input.
            2. **POS Tagging**: Identifies parts of speech (Nouns, Verbs, etc.) using `AfroXLMR`.
            3. **Tone Injection**: Assigns tones based on POS tags (e.g., Verbs -> High Tone).
            4. **G2P Conversion**: Converts text to IPA phonemes.
            5. **Synthesis**: VITS model generates audio from phonemes.
            """
        )
