import os
import sys
import gradio as gr
import shutil
import subprocess
import tempfile
import torch
import torchaudio
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
import configparser
from easy_functions import get_video_details

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Load ASR pipeline
pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3",
    torch_dtype=torch.float16,
    device=device,
)

# Load voice cloning model
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# Settings for voice cloning
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

# Inferencing Logic for Voice Cloning
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

# Lip Sync Function
def lip_sync(video_file, vocal_file, output_suffix="", preview_settings=False,
             quality="fast", output_height="full resolution", wav2lip_version="Wav2Lip",
             nosmooth=False, U=0, D=0, L=0, R=0):
    
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

# Gradio Interface
def interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Voice Cloning and Lip Sync Application")

        # Step 1: Voice Cloning
        with gr.Row():
            with gr.Column(scale=1):
                ref_audio_input = gr.Audio(label="Step 1: Upload Reference Audio", type="filepath")
                ref_text_input = gr.Textbox(label="Reference Text (Optional)", info="Leave blank for automatic transcription.", lines=2)
                gen_text_input = gr.Textbox(label="Step 2: Enter Text to Generate", lines=5)
                remove_silence = gr.Checkbox(label="Remove Silences", value=True)
                generate_btn = gr.Button("Generate Cloned Audio")

            audio_output = gr.Audio(label="Generated Cloned Audio")

        # Step 2: Lip Sync Option
        with gr.Row():
            lip_sync_option = gr.Checkbox(label="Do you want to perform lip sync?", value=False)
            video_input = gr.File(label="Upload Video File (.mp4)", visible=False)
            audio_input = gr.File(label="Upload Audio File (.wav)", visible=False)
            output_suffix = gr.Textbox(label="Output Suffix", placeholder="Enter optional suffix for output filename", visible=False)
            process_btn = gr.Button("Process Lip Sync", visible=False)

        # Step 3: Output Video
        output_video = gr.Video(label="Output Lip-Synced Video", visible=False)
        result_message = gr.Textbox(label="Processing Result", visible=False)

        # Voice Cloning Logic
        def generate_cloned_audio(ref_audio, ref_text, gen_text, remove_silence):
            (sample_rate, generated_wave), spectrogram_path = infer(ref_audio, ref_text, gen_text, remove_silence)
            return (sample_rate, generated_wave), spectrogram_path

        generate_btn.click(
            fn=generate_cloned_audio,
            inputs=[ref_audio_input, ref_text_input, gen_text_input, remove_silence],
            outputs=[audio_output]
        )

        # Lip Sync Logic
        def toggle_lip_sync(checked):
            video_input.visible = checked
            audio_input.visible = checked
            output_suffix.visible = checked
            process_btn.visible = checked

        lip_sync_option.change(toggle_lip_sync, inputs=lip_sync_option)

        def process_lip_sync(video_file, audio_file, output_suffix):
            video_path = video_file.name if video_file else ""
            audio_path = audio_file.name if audio_file else ""
            result_path, message = lip_sync(video_path, audio_path, output_suffix)
            return result_path, message

        process_btn.click(
            fn=process_lip_sync,
            inputs=[video_input, audio_input, output_suffix],
            outputs=[output_video, result_message]
        )

    return demo

# Launch the Gradio interface
if __name__ == "__main__":
    gradio_interface = interface()
    gradio_interface.launch(share=True)