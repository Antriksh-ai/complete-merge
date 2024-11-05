import os
import sys
import gradio as gr
import shutil
import subprocess
import configparser
from easy_functions import get_video_details
import torch
import torchaudio
import tempfile
from vocos import Vocos
from pydub import AudioSegment, silence
from model import CFM, UNetT
from cached_path import cached_path
from model.utils import load_checkpoint, get_tokenizer, save_spectrogram
from transformers import pipeline
import soundfile as sf

# Set up device for torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Load ASR and vocoder pipelines
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device=device)
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# Load voice cloning model
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256

def load_voice_model():
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    model = CFM(
        transformer=UNetT(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(target_sample_rate=target_sample_rate, n_mel_channels=n_mel_channels, hop_length=hop_length),
        vocab_char_map=vocab_char_map,
    ).to(device)
    model = load_checkpoint(model, ckpt_path, device, use_ema=True)
    return model

model = load_voice_model()

# Lip-sync function
def lip_sync(video_file, vocal_file, output_suffix="", preview_settings=False, quality="fast", output_height="full resolution",
             wav2lip_version="Wav2Lip", nosmooth=False, U=0, D=0, L=0, R=0):
    
    config = configparser.ConfigParser()
    config.read('config.ini')
    working_directory = os.getcwd()
    
    temp_folder = os.path.join(working_directory, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    if not video_file or not os.path.exists(video_file):
        return None, "Error: Video file not found. Please upload a valid file."
    if vocal_file and not os.path.exists(vocal_file):
        return None, "Error: Audio file not found. Please upload a valid file."

    checkpoint_path = os.path.join(working_directory, "checkpoints", wav2lip_version + ".pth")
    if not os.path.exists(checkpoint_path):
        return None, f"Error: Wav2Lip checkpoint '{checkpoint_path}' not found."

    resolution_scale = {"half resolution": 2, "full resolution": 1}.get(output_height, 1)
    in_width, in_height, _, _ = get_video_details(video_file)
    out_height = round(in_height / resolution_scale)

    temp_output = os.path.join(temp_folder, "output.mp4")
    output_filename = os.path.splitext(os.path.basename(video_file))[0] + output_suffix + ".mp4"
    output_video = os.path.join(working_directory, output_filename)

    cmd = [
        sys.executable, "inference.py", "--face", video_file, "--audio", vocal_file,
        "--outfile", temp_output, "--pads", str(U), str(D), str(L), str(R),
        "--checkpoint_path", checkpoint_path, "--out_height", str(out_height),
        "--fullres", str(resolution_scale), "--quality", quality,
        "--nosmooth", str(int(nosmooth)), "--preview_settings", str(preview_settings)
    ]

    process_result = subprocess.run(cmd, capture_output=True, text=True)

    if os.path.exists(temp_output):
        shutil.move(temp_output, output_video)
        shutil.rmtree(temp_folder, ignore_errors=True)
        return output_video, "Lip-sync processing completed successfully!"
    else:
        return None, f"Processing failed: {process_result.stderr}"

# Voice cloning function
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
        ref_text = pipe(ref_audio, chunk_length_s=30, batch_size=128, generate_kwargs={"task": "transcribe"}, return_timestamps=False)["text"].strip()
    
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

    progress(100, desc="Done!")
    return (target_sample_rate, generated_wave), None

custom_css = """
/* Custom CSS for Dark Theme */
:root { --background-fill-primary: #1a1a1a !important; --background-fill-secondary: #2d2d2d !important; --border-color-primary: #404040 !important; --text-color: #ffffff !important; --color-accent-soft: #3d4c7d !important; }
"""

custom_theme = gr.themes.Soft(
    primary_hue="indigo", secondary_hue="slate", neutral_hue="slate", font=gr.themes.GoogleFont("Inter")
).set(
    body_background_fill="#1a1a1a", body_text_color="#ffffff", background_fill_primary="#2d2d2d", border_color_primary="#404040"
)

# Gradio App
with gr.Blocks(theme=custom_theme, css=custom_css) as app:
    with gr.Tab("Lip Sync"):
        gr.Markdown("# Wav2Lip Lip Sync Application")
        video_input = gr.File(label="Upload Video File (.mp4)")
        audio_input = gr.File(label="Upload Audio File (.wav)")
        output_suffix = gr.Textbox(label="Output Suffix", placeholder="Optional output filename suffix")
        preview_settings = gr.Checkbox(label="Preview Settings", value=False)
        quality = gr.Radio(["fast", "high"], label="Processing Quality", value="fast")
        output_height = gr.Radio(["half resolution", "full resolution"], label="Output Resolution", value="full resolution")
        wav2lip_version = gr.Radio(["Wav2Lip", "Wav2Lip_GAN"], label="Wav2Lip Version", value="Wav2Lip")
        nosmooth = gr.Checkbox(label="No Smoothing", value=False)
        U = gr.Slider(label="Padding Top", minimum=0, maximum=50, step=1, value=0)
        D = gr.Slider(label="Padding Bottom", minimum=0, maximum=50, step=1, value=0)
        L = gr.Slider(label="Padding Left", minimum=0, maximum=50, step=1, value=0)
        R = gr.Slider(label="Padding Right", minimum=0, maximum=50, step=1, value=0)
        output_video = gr.Video(label="Output Video", autoplay=True)
        result = gr.Textbox(label="Processing Result")

        def process_lip_sync(video_file, audio_file, output_suffix, preview_settings, quality, output_height, wav2lip_version, nosmooth, U, D, L, R):
            video_path = video_file.name if video_file else ""
            audio_path = audio_file.name if audio_file else ""
            result_path, message = lip_sync(video_path, audio_path, output_suffix, preview_settings, quality, output_height, wav2lip_version, nosmooth, U, D, L, R)
            return (result_path, message) if result_path else (None, message)

        submit_button = gr.Button("Process")
        submit_button.click(process_lip_sync, inputs=[video_input, audio_input, output_suffix, preview_settings, quality, output_height, wav2lip_version, nosmooth, U, D, L, R], outputs=[output_video, result])

    with gr.Tab("Voice Cloning"):
        gr.Markdown("# Antriksh AI - Voice Cloning Application")
        ref_audio_input = gr.Audio(label="Upload Reference Audio", type="filepath")
        gen_text_input = gr.Textbox(label="Enter Text to Generate", lines=5)
        ref_text_input = gr.Textbox(label="Reference Text (Optional)", info="Leave blank for automatic transcription.", lines=2)
        remove_silence = gr.Checkbox(label="Remove Silences", value=True)
        audio_output = gr.Audio(label="Generated Audio")
        spectrogram_output = gr.Image(label="Spectrogram")

        generate_btn = gr.Button("Synthesize")
        generate_btn.click(infer, inputs=[ref_audio_input, ref_text_input, gen_text_input, remove_silence], outputs=[audio_output, spectrogram_output])

if __name__ == "__main__":
    app.launch(share=True)
