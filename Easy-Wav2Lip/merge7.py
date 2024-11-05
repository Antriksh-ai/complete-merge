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
import os
import subprocess

# Device setting
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize ASR pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device=device,
)

# Load vocos model for mel spectrogram decoding
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# Voice cloning settings
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0

# Load voice cloning model
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

# Function for voice cloning
def clone_voice(ref_audio, ref_text, gen_text, remove_silence, progress=gr.Progress()):
    if not ref_audio:
        return None, "No reference audio provided. Please upload a valid audio file."
    
    progress(0, desc="Processing audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio)
        if remove_silence:
            non_silent_segs = silence.split_on_silence(
                aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000
            )
            non_silent_wave = AudioSegment.silent(duration=0)
            for seg in non_silent_segs:
                non_silent_wave += seg
            aseg = non_silent_wave
        audio_duration = len(aseg)
        if audio_duration > 15000:
            gr.Warning("Audio is over 15s, clipping to only first 15s.")
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio_path = f.name

    progress(20, desc="Transcribing audio...")
    if not ref_text.strip():
        ref_text = pipe(
            ref_audio_path,
            chunk_length_s=30,
            batch_size=128,
            generate_kwargs={"task": "transcribe"},
            return_timestamps=False,
        )["text"].strip()
    
    if not ref_text.endswith(". "):
        ref_text += ". " if not ref_text.endswith(".") else " "

    progress(40, desc="Generating audio...")
    audio, sr = torchaudio.load(ref_audio_path)
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

    cloned_audio_path = tempfile.mktemp(suffix=".wav")
    sf.write(cloned_audio_path, generated_wave, target_sample_rate)

    spectrogram_path = tempfile.mktemp(suffix=".png")
    save_spectrogram(generated_mel_spec[0].cpu().numpy(), spectrogram_path)

    progress(100, desc="Done!")
    return cloned_audio_path, spectrogram_path

# Lip-sync function
def lip_sync(video_file, vocal_audio):
    # Ensure video_file and vocal_audio are strings
    if not isinstance(video_file, str) or not isinstance(vocal_audio, str):
        return None, "Invalid file paths provided for video or audio."
    
    print(f"Video file path: {video_file}")
    print(f"Audio file path: {vocal_audio}")

    working_directory = os.getcwd()
    temp_output = os.path.join(working_directory, "temp_output.mp4")
    checkpoint_path = os.path.join(working_directory, "checkpoints", "Wav2Lip.pth")
    
    if not os.path.exists(checkpoint_path):
        return None, "Wav2Lip checkpoint not found."

    cmd = [
        "python", "inference.py", "--face", video_file, "--audio", vocal_audio,
        "--outfile", temp_output, "--checkpoint_path", checkpoint_path
    ]

    process_result = subprocess.run(cmd, capture_output=True, text=True)

    if os.path.exists(temp_output):
        return temp_output, "Lip-sync processing completed successfully!"
    else:
        return None, f"Processing failed: {process_result.stderr}"

# Gradio interface setup
with gr.Blocks() as app:
    gr.Markdown("# Voice Cloning and Lip-Sync Application")

    with gr.Row():
        ref_audio_input = gr.Audio(label="Step 1: Upload Reference Audio (for voice cloning)", type="filepath")
        gen_text_input = gr.Textbox(label="Step 2: Enter Text to Generate in Cloned Voice", lines=5)
        remove_silence = gr.Checkbox(label="Remove Silences from Cloned Audio", value=True)
        synthesize_btn = gr.Button("Synthesize Voice")

    with gr.Row():
        audio_output = gr.Audio(label="Generated Audio")
        spectrogram_output = gr.Image(label="Spectrogram")

    def on_voice_generated(ref_audio, ref_text, gen_text, remove_silence):
        cloned_audio, spectrogram = clone_voice(ref_audio, ref_text, gen_text, remove_silence)
        return cloned_audio, spectrogram

    synthesize_btn.click(
        on_voice_generated,
        inputs=[ref_audio_input, gr.Textbox(label="Reference Text (Leave blank for auto transcription)"), gen_text_input, remove_silence],
        outputs=[audio_output, spectrogram_output],
    )

    with gr.Accordion("Optional Lip-Sync Step"):
        lip_sync_prompt = gr.Checkbox(label="Do you want to lip-sync this audio to a video?")
        video_input = gr.File(label="Upload Video File (.mp4)")

        def handle_lip_sync(should_lip_sync, video_file, cloned_audio):
            if should_lip_sync:
                video_path = video_file.name if video_file else ""
                audio_path = cloned_audio if isinstance(cloned_audio, str) else ""
                lip_synced_video, message = lip_sync(video_path, audio_path)
                return lip_synced_video, message
            return None, "Lip-sync not performed."

        lip_sync_button = gr.Button("Process Lip-Sync")
        lip_sync_button.click(
            handle_lip_sync,
            inputs=[lip_sync_prompt, video_input, audio_output],
            outputs=[gr.Video(label="Lip-Synced Video"), gr.Textbox(label="Lip-Sync Status")],
        )

app.launch(share=True)
