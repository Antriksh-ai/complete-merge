import os
import gradio as gr
import torch
import torchaudio
import tempfile
import shutil
import subprocess
import configparser
from pydub import AudioSegment, silence
from vocos import Vocos
from model import CFM, UNetT
from cached_path import cached_path
from model.utils import load_checkpoint, get_tokenizer, save_spectrogram
from transformers import pipeline
import soundfile as sf
from easy_functions import get_video_details

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Initialize ASR and vocoder models
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float16, device=device)
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# Settings for audio processing
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0

# Load TTS model
def load_model():
    model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.safetensors"))
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    model = CFM(
        transformer=UNetT(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(target_sample_rate=target_sample_rate, n_mel_channels=n_mel_channels, hop_length=hop_length),
        odeint_kwargs=dict(method=ode_method),
        vocab_char_map=vocab_char_map,
    ).to(device)
    model = load_checkpoint(model, ckpt_path, device, use_ema=True)
    return model

model = load_model()

# Voice Cloning Function
def generate_cloned_voice(ref_audio, ref_text, gen_text, remove_silence, progress=gr.Progress()):
    progress(0, desc="Processing audio...")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        aseg = AudioSegment.from_file(ref_audio)
        non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000)
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            non_silent_wave += non_silent_seg
        aseg = non_silent_wave
        if len(aseg) > 15000:
            gr.Warning("Audio is over 15s, clipping to only first 15s.")
            aseg = aseg[:15000]
        aseg.export(f.name, format="wav")
        ref_audio = f.name

    progress(20, desc="Transcribing audio...")
    if not ref_text.strip():
        ref_text = pipe(ref_audio, chunk_length_s=30, batch_size=128, generate_kwargs={"task": "transcribe"}, return_timestamps=False)["text"].strip()
    
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
        generated, _ = model.sample(cond=audio, text=text_list, duration=duration, steps=nfe_step, cfg_strength=cfg_strength, sway_sampling_coef=sway_sampling_coef)
    generated = generated[:, audio.shape[-1] // hop_length:, :]
    generated_mel_spec = generated.permute(0, 2, 1)
    generated_wave = vocos.decode(generated_mel_spec.cpu())
    generated_wave = generated_wave.squeeze().cpu().numpy()

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

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_audio_file:
        sf.write(output_audio_file.name, generated_wave, target_sample_rate)
        return output_audio_file.name

# Lip Sync Function
def lip_sync(video_file, vocal_file, output_suffix="", quality="high", output_height="full resolution"):
    working_directory = os.getcwd()
    temp_folder = os.path.join(working_directory, "temp")
    os.makedirs(temp_folder, exist_ok=True)
    checkpoint_path = os.path.join(working_directory, "checkpoints", "Wav2Lip_GAN.pth")
    resolution_scale = {"half resolution": 2, "full resolution": 1}.get(output_height, 1)
    in_width, in_height, _, _ = get_video_details(video_file)
    out_height = round(in_height / resolution_scale)

    temp_output = os.path.join(temp_folder, "output.mp4")
    output_filename = os.path.splitext(os.path.basename(video_file))[0] + output_suffix + ".mp4"
    output_video = os.path.join(working_directory, output_filename)

    cmd = [
        "python3", "inference.py", "--face", video_file, "--audio", vocal_file,
        "--outfile", temp_output, "--checkpoint_path", checkpoint_path, "--out_height", str(out_height),
        "--fullres", str(resolution_scale), "--quality", quality
    ]
    subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(temp_output):
        shutil.move(temp_output, output_video)
        shutil.rmtree(temp_folder, ignore_errors=True)
        return output_video
    return None

# Gradio Interface
# Gradio Interface
# Adjusted Gradio Interface for Separate Lip Sync Function
def interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Voice Cloning and Lip Sync Application")

        with gr.Row():
            ref_audio_input = gr.Audio(label="Step 1: Upload Reference Audio", type="filepath")
            ref_text_input = gr.Textbox(label="Reference Text (Optional)", placeholder="Leave blank for auto transcription")
            gen_text_input = gr.Textbox(label="Enter Text to Generate in Cloned Voice")
            remove_silence = gr.Checkbox(label="Remove Silence from Cloned Voice", value=True)

        with gr.Row():
            clone_button = gr.Button("Generate Cloned Voice")
            cloned_audio_output = gr.Audio(label="Cloned Voice Output", type="filepath")

        with gr.Row():
            lip_sync_option = gr.Checkbox(label="Apply Lip Sync to Video", value=False)
            video_input = gr.File(label="Upload Video File (.mp4)", visible=False)
            lip_sync_button = gr.Button("Start Lip Sync", visible=False)
            lip_sync_output = gr.Video(label="Lip Synced Video Output", visible=False)

        # Function to show/hide video input and lip-sync button based on lip_sync_option
        def show_video_input(lip_sync):
            return (
                gr.update(visible=lip_sync),
                gr.update(visible=lip_sync),
                gr.update(visible=lip_sync)
            )

        # Generate cloned voice only
        def generate_cloned_voice_only(ref_audio, ref_text, gen_text, remove_silence):
            cloned_audio_path = generate_cloned_voice(ref_audio, ref_text, gen_text, remove_silence)
            return cloned_audio_path

        # Generate lip-synced video using the cloned voice and video input
        def generate_lip_sync_video(cloned_audio_path, video_file):
            if video_file:
                lip_synced_video_path = lip_sync(video_file, cloned_audio_path)
                return lip_synced_video_path

        lip_sync_option.change(show_video_input, inputs=lip_sync_option, outputs=[video_input, lip_sync_button, lip_sync_output])

        clone_button.click(
            fn=generate_cloned_voice_only,
            inputs=[ref_audio_input, ref_text_input, gen_text_input, remove_silence],
            outputs=[cloned_audio_output]
        )

        lip_sync_button.click(
            fn=generate_lip_sync_video,
            inputs=[cloned_audio_output, video_input],
            outputs=[lip_sync_output]
        )

    return demo

if __name__ == "__main__":
    gradio_interface = interface()
    gradio_interface.launch(share=True)

