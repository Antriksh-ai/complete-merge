import os
import sys
import gradio as gr
import shutil
import subprocess
import configparser
from easy_functions import get_video_details

# Define the main processing function
def lip_sync(video_file, vocal_file, output_suffix="", preview_settings=False,
             quality="fast", output_height="full resolution", wav2lip_version="Wav2Lip",
             nosmooth=False, U=0, D=0, L=0, R=0):
    
    # Configurations and initial settings
    config = configparser.ConfigParser()
    config.read('config.ini')
    working_directory = os.getcwd()
    
    # Ensure temp folder exists
    temp_folder = os.path.join(working_directory, "temp")
    os.makedirs(temp_folder, exist_ok=True)

    # Verify video and audio file paths
    if not video_file or not os.path.exists(video_file):
        return None, "Error: Video file not found. Please upload a valid file."
    if vocal_file and not os.path.exists(vocal_file):
        return None, "Error: Audio file not found. Please upload a valid file."

    # Determine the Wav2Lip checkpoint path based on version
    checkpoint_path = os.path.join(working_directory, "checkpoints", wav2lip_version + ".pth")
    if not os.path.exists(checkpoint_path):
        return None, f"Error: Wav2Lip checkpoint '{checkpoint_path}' not found."

    # Set resolution scale and output dimensions
    resolution_scale = {"half resolution": 2, "full resolution": 1}.get(output_height, 1)
    in_width, in_height, _, _ = get_video_details(video_file)
    out_height = round(in_height / resolution_scale)

    # Define temporary and output video paths
    temp_output = os.path.join(temp_folder, "output.mp4")
    output_filename = os.path.splitext(os.path.basename(video_file))[0] + output_suffix + ".mp4"
    output_video = os.path.join(working_directory, output_filename)

    # Prepare the command to execute Wav2Lip processing
    cmd = [
        sys.executable, "inference.py", "--face", video_file, "--audio", vocal_file,
        "--outfile", temp_output, "--pads", str(U), str(D), str(L), str(R),
        "--checkpoint_path", checkpoint_path, "--out_height", str(out_height),
        "--fullres", str(resolution_scale), "--quality", quality,
        "--nosmooth", str(int(nosmooth)), "--preview_settings", str(preview_settings)
    ]

    # Run the Wav2Lip command
    process_result = subprocess.run(cmd, capture_output=True, text=True)

    # Check if processing was successful by confirming output file creation
    if os.path.exists(temp_output):
        shutil.move(temp_output, output_video)  # Move the output to a permanent location
        shutil.rmtree(temp_folder, ignore_errors=True)  # Clean up temp files
        return output_video, "Lip-sync processing completed successfully!"
    else:
        # Return the error message if the processing failed
        return None, f"Processing failed: {process_result.stderr}"

# Define Gradio Interface
def interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Wav2Lip Lip Sync Application")

        with gr.Row():
            video_input = gr.File(label="Upload Video File (.mp4)")
            audio_input = gr.File(label="Upload Audio File (.wav)")

        with gr.Row():
            output_suffix = gr.Textbox(label="Output Suffix", placeholder="Enter optional suffix for output filename")
            preview_settings = gr.Checkbox(label="Preview Settings", value=False)

        with gr.Row():
            quality = gr.Radio(["fast", "high"], label="Processing Quality", value="fast")
            output_height = gr.Radio(["half resolution", "full resolution"], label="Output Resolution", value="full resolution")
            wav2lip_version = gr.Radio(["Wav2Lip", "Wav2Lip_GAN"], label="Wav2Lip Version", value="Wav2Lip")

        with gr.Row():
            nosmooth = gr.Checkbox(label="No Smoothing", value=False)
            U = gr.Slider(label="Padding Top", minimum=0, maximum=50, step=1, value=0)
            D = gr.Slider(label="Padding Bottom", minimum=0, maximum=50, step=1, value=0)
            L = gr.Slider(label="Padding Left", minimum=0, maximum=50, step=1, value=0)
            R = gr.Slider(label="Padding Right", minimum=0, maximum=50, step=1, value=0)

        output_video = gr.Video(label="Output Video", autoplay=True)
        result = gr.Textbox(label="Processing Result")

        # Function to process files and return output video path or error
        def process_files(video_file, audio_file, output_suffix, preview_settings, quality,
                          output_height, wav2lip_version, nosmooth, U, D, L, R):
            video_path = video_file.name if video_file else ""
            audio_path = audio_file.name if audio_file else ""
            result_path, message = lip_sync(video_path, audio_path, output_suffix, preview_settings,
                                            quality, output_height, wav2lip_version, nosmooth, U, D, L, R)
            return (result_path, message) if result_path else (None, message)

        submit_button = gr.Button("Process")
        submit_button.click(
            fn=process_files,
            inputs=[video_input, audio_input, output_suffix, preview_settings, quality,
                    output_height, wav2lip_version, nosmooth, U, D, L, R],
            outputs=[output_video, result]
        )

    return demo

# Launch the Gradio interface
if __name__ == "__main__":
    gradio_interface = interface()
    gradio_interface.launch(share=True)
