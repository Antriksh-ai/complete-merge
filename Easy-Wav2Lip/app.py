# Gradio Application for Voice Cloning
# Version as of 21/10/2024

import gradio as gr
import torch
import torchaudio
import tempfile
from vocos import Vocos
from pydub import AudioSegment, silence
from model import CFM, UNetT
from cached_path import cached_path
from model.utils import (
    load_checkpoint,
    get_tokenizer,
    save_spectrogram,
)
from transformers import pipeline
import soundfile as sf

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device=device,
)
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# Settings
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0

def load_model():
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    model = CFM(
        transformer=UNetT(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)
    model = load_checkpoint(model, ckpt_path, device, use_ema=True)
    return model

model = load_model()

# Inferencing Logic

def infer(ref_audio, ref_text, gen_text, remove_silence, progress=gr.Progress()):
    progress(0, desc="Processing audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio)
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        aseg = non_silent_wave
        audio_duration = len(aseg)
        if audio_duration > 15000:
            gr.Warning("Audio is over 15s, clipping to only first 15s.")
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    progress(20, desc="Transcribing audio...")
    if not ref_text.strip():
        ref_text = pipe(
            ref_audio,
            chunk_length_s=30,
            batch_size=128,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )["text"].strip()
    
    if not ref_text.endswith(". "):
        ref_text += ". " if not ref_text.endswith(".") else " "

    progress(40, desc="Generating audio...")
    audio, sr = torchaudio.load(ref_audio)
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)
    audio = audio.to(device)

    text_list = [ref_text + gen_text]
    duration = audio.shape[-1] // hop_length + int(audio.shape[-1] / hop_length / len(ref_text) * len(gen_text) / speed)

    progress(60, desc="Synthesizing speech...")
    with torch.inference_mode():
        generated, _ = model.sample(
            cond=audio,
            text=text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
        )

    generated = generated.to(torch.float32)
    generated = generated[:, audio.shape[-1] // hop_length:, :]
    generated_mel_spec = generated.permute(0, 2, 1)
    generated_wave = vocos.decode(generated_mel_spec.cpu())
    if rms < target_rms:
        generated_wave = generated_wave * rms / target_rms

    generated_wave = generated_wave.squeeze().cpu().numpy()

    progress(80, desc="Post-processing...")
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, generated_wave, target_sample_rate)
            aseg = AudioSegment.from_file(f.name)
            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave
            aseg.export(f.name, format="wav")
            generated_wave, _ = torchaudio.load(f.name)
        generated_wave = generated_wave.squeeze().cpu().numpy()

    progress(90, desc="Generating spectrogram...")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(generated_mel_spec[0].cpu().numpy(), spectrogram_path)

    progress(100, desc="Done!")
    return (target_sample_rate, generated_wave), spectrogram_path

custom_css = """

/* Dark theme customization */
:root {
    --background-fill-primary: #1a1a1a !important;
    --background-fill-secondary: #2d2d2d !important;
    --border-color-primary: #404040 !important;
    --text-color: #ffffff !important;
    --body-text-color: #ffffff !important;
    --color-accent-soft: #3d4c7d !important;
}

body {
    background-color: #1a1a1a !important;
    color: #ffffff !important;
}

.gradio-container {
    background-color: #1a1a1a !important;
}

.tabs {
    background-color: #2d2d2d !important;
}

.tab-selected {
    background-color: #404040 !important;
}

#logo-column {
    display: flex;
    justify-content: flex-end;
    align-items: flex-start;
    background-color: transparent !important;
}

#logo-column img {
    max-width: 180px;
    height: auto;
    margin-top: 10px;
    filter: brightness(0.9);
}

.gr-box {
    background-color: #2d2d2d !important;
    border: 1px solid #404040 !important;
}

.gr-button {
    background-color: #3d4c7d !important;
    color: white !important;
}

.gr-button:hover {
    background-color: #4a5d99 !important;
}

/* Modified input styling for darker background */
.gr-input, .gr-textarea {
    background-color: #1a1a1a !important;
    color: white !important;
    border: 1px solid #404040 !important;
}

#step-2-input textarea {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-color: #404040 !important;
}

#step-2-input textarea:focus {
    border-color: #3d4c7d !important;
    box-shadow: 0 0 0 2px rgba(61, 76, 125, 0.2) !important;
}

#reference-text-input textarea {
    background-color: #fffff !important;
    color: #000000!important;
    border-color: #404040 !important;
}

#reference-text-input textarea:focus {
    border-color: #3d4c7d !important;
    box-shadow: 0 0 0 2px rgba(61, 76, 125, 0.2) !important;
}

.gr-accordion {
    background-color: #2d2d2d !important;
}

.gr-form {
    background-color: transparent !important;
}

.markdown-text {
    color: #ffffff !important;
}

.markdown-text h1, .markdown-text h2, .markdown-text h3 {
    color: #ffffff !important;
}

.audio-player {
    background-color: #2d2d2d !important;
    border: 1px solid #404040 !important;
}

"""

custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=gr.themes.GoogleFont("Inter"),
).set(
    body_background_fill="#1a1a1a",
    body_background_fill_dark="#1a1a1a",
    body_text_color="#ffffff",
    body_text_color_dark="#ffffff",
    background_fill_primary="#2d2d2d",
    background_fill_primary_dark="#2d2d2d",
    background_fill_secondary="#1a1a1a",
    background_fill_secondary_dark="#1a1a1a",
    border_color_primary="#404040",
    border_color_primary_dark="#404040",
    button_primary_background_fill="#3d4c7d",
    button_primary_background_fill_dark="#3d4c7d",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
)


with gr.Blocks(theme=custom_theme, css=custom_css) as app:
    
    with gr.Row():
        with gr.Column(scale=9):
            gr.Markdown(
                """
                # Antriksh AI

                Welcome to our voice cloning application! Follow these steps to create your own custom voice:

                1. Upload a short audio clip (less than 15 seconds) of the voice you want to clone.
                2. Enter the text you want to generate in the new voice.
                3. Click "Synthesize" and listen to hear the magic!

                It's that easy! Let's get started.
                """
            )
            
        with gr.Column(scale=1, elem_id="logo-column"):
            gr.Image("logo/logo-removebg-preview.png", label="", show_label=False)
    
    with gr.Row():
        with gr.Column(scale=1):
            ref_audio_input = gr.Audio(
                label="Step 1: Upload Reference Audio",
                type="filepath",
                elem_classes="audio-player"
            )
            gen_text_input = gr.Textbox(
                label="Step 2: Enter Text to Generate",
                lines=5,
                elem_id="step-2-input",
                elem_classes="gr-textarea"
            )
            generate_btn = gr.Button(
                "Step 3: Synthesize",
                variant="primary",
                elem_classes="gr-button"
            )

        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Audio",
                elem_classes="audio-player"
            )
            spectrogram_output = gr.Image(label="Spectrogram")

    with gr.TabItem("Advanced Settings"):
        gr.Markdown(
            "These settings are optional. If you're not sure, leave them as they are."
        )
        ref_text_input = gr.Textbox(
            label="Reference Text (Optional)",
            info="Leave blank for automatic transcription.",
            lines=2,
            elem_id="reference-text-input",
            elem_classes="gr-textarea"
        )
        remove_silence = gr.Checkbox(
            label="Remove Silences",
            info="This can improve the quality of longer audio clips.",
            value=True
        )

    generate_btn.click(
        infer,
        inputs=[
            ref_audio_input,
            ref_text_input,
            gen_text_input,
            remove_silence,
        ],
        outputs=[audio_output, spectrogram_output],
    )
    
    with gr.TabItem("How It Works"):
        gr.Markdown(
            """
            # How Voice Cloning Works

            Our voice cloning system uses advanced AI technology to create a synthetic voice that sounds like the reference audio you provide. Here's a simplified explanation of the process:

            1. **Audio Analysis**: When you upload a reference audio clip, our system analyzes its unique characteristics, including pitch, tone, and speech patterns.

            2. **Text Processing**: The text you want to generate is processed and converted into a format that our AI model can understand.

            3. **Voice Synthesis**: Our AI model, based on the E2-TTS (Embarrassingly Easy Text-to-Speech) architecture, combines the characteristics of the reference audio with the new text to generate a synthetic voice.

            4. **Audio Generation**: The synthetic voice is converted into an audio waveform, which you can then play back or download.

            5. **Spectrogram Creation**: A visual representation of the audio (spectrogram) is generated, showing the frequency content of the sound over time.

            This process allows you to generate new speech in the voice of the reference audio, even saying things that weren't in the original recording. It's a powerful tool for creating custom voiceovers, audiobooks, or just for fun!

            Remember, the quality of the output depends on the quality and length of the input audio. For best results, use a clear, high-quality audio clip of 10-15 seconds in length.
            """
        )

if __name__ == "__main__":
    app.launch(share=True)