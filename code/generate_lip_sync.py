import os
import torch
from TTS.api import TTS
import sys
import subprocess 
import argparse

def check_file_exists(file_path, description):
    if not os.path.exists(file_path):
        print(f"[ERROR] {description} not found: {file_path}")
        sys.exit(1)

def generate_speech(input_text_file, speaker_wav, output_wav, language='en', use_cuda=True):
    print("[INFO] Loading TTS model...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to("cuda" if use_cuda else "cpu")
    print("[INFO] TTS model loaded successfully.")

    # Read input text
    print(f"[INFO] Reading text from {input_text_file}...")
    with open(input_text_file, 'r') as file:
        text = file.read().strip()
    print(f"[INFO] Text to synthesize: {text}")

    # Generate speech
    print(f"[INFO] Generating speech using {speaker_wav} as speaker reference...")
    tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=output_wav)
    print(f"[INFO] Speech synthesis complete. Output saved to {output_wav}.")

#change run_wav2lip to be inside inference folder 
def run_wav2lip(face_video, audio_wav, output_video):
    print("[INFO] Running Wav2Lip using subprocess...")

    checkpoint_path = os.path.expanduser("~/.wav2lip/checkpoints/wav2lip_gan.pth")

    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
        return

    command = [
        "python3", "inference.py",
        "--checkpoint_path", checkpoint_path,
        "--face", face_video,
        "--audio", audio_wav,
        "--outfile", output_video
    ]

    print(f"[DEBUG] Executing command: {' '.join(command)}")

    result = subprocess.run(command, capture_output=True, text=True, cwd=os.path.expanduser("~/Wav2Lip"))

    if result.returncode == 0:
        print(f"[INFO] Wav2Lip processing complete. Output saved to {output_video}.")
    else:
        print(f"[ERROR] Wav2Lip failed with error:\n{result.stderr}")


# def run_wav2lip(face_video, audio_wav, output_video):
#     print("[INFO] Running Wav2Lip using subprocess...")

#     checkpoint_path = os.path.expanduser("~/.wav2lip/checkpoints/wav2lip_gan.pth")

#     if not os.path.exists(checkpoint_path):
#         print(f"[ERROR] Checkpoint file not found: {checkpoint_path}")
#         return

#     command = [
#         "python3", os.path.expanduser("~/Wav2Lip/inference.py"),
#         "--checkpoint_path", checkpoint_path,
#         "--face", face_video,
#         "--audio", audio_wav,
#         "--outfile", output_video
#     ]

#     print(f"[DEBUG] Executing command: {' '.join(command)}")

#     result = subprocess.run(command, capture_output=True, text=True)

#     if result.returncode == 0:
#         print(f"[INFO] Wav2Lip processing complete. Output saved to {output_video}.")
#     else:
#         print(f"[ERROR] Wav2Lip failed with error:\n{result.stderr}")


def main():
    # commented out the manual method that didn't take inputs 
    # print("[INFO] Starting lip-sync script...")
    # input_text_file = "../input_script_0.txt"
    # speaker_wav = "../inputs/audio_input_files/combined_samples.wav"
    # output_wav = "../outputs/combined_speaker_script0/00_XTTS_v2_combined_speaker_script_0.wav"
    # face_video = "../inputs/video_input_files/hey_gen_2.mp4"
    # output_video = "output.mp4"

    # print("[INFO] Checking input files...")
    # check_file_exists(input_text_file, "Input script")
    # check_file_exists(speaker_wav, "Speaker audio sample")
    # check_file_exists(face_video, "Face video file")
    # print("[INFO] Generating speech...")
    # generate_speech(input_text_file, speaker_wav, output_wav)

    # print("[INFO] Running Wav2Lip inference...")
    # run_wav2lip(face_video, output_wav, output_video)

    # print(f"[INFO] Lip-synced video saved as {output_video}")
    # print("[INFO] Process completed successfully.")

    ## this version takes inputs 

    print("[INFO] Starting lip-sync script...")

    parser = argparse.ArgumentParser(description="Lip-sync script with default values and optional inputs.")
    parser.add_argument("--text", default="../input_script_0.txt", help="Path to input text file")
    parser.add_argument("--audio", default="../inputs/audio_input_files/combined_samples.wav", help="Path to speaker WAV file")
    parser.add_argument("--face", default="../inputs/video_input_files/hey_gen_2.mp4", help="Path to face video file")
    parser.add_argument("--output_audio", default="../outputs/combined_speaker_script0/00_XTTS_v2_combined_speaker_script_0.wav", help="Path to output WAV file")
    parser.add_argument("--output_video", default="output.mp4", help="Path to output video file")

    args = parser.parse_args()

    print("[INFO] Starting lip-sync script...")
    
    print("[INFO] Checking input files...")
    check_file_exists(args.text, "Input script")
    check_file_exists(args.audio, "Speaker audio sample")
    check_file_exists(args.face, "Face video file")

    print("[INFO] Generating speech...")
    generate_speech(args.text, args.audio, args.output_audio)

    print("[INFO] Running Wav2Lip inference...")
    run_wav2lip(args.face, args.output_audio, args.output_video)

    print(f"[INFO] Lip-synced video saved as {args.output_video}")
    print("[INFO] Process completed successfully.")

if __name__ == "__main__":
    main()
