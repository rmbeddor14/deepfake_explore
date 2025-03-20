# ## just an api wrapper for generate_lip_sync.py that can take from the UI and spit out a video
# from flask import Flask, render_template, request, jsonify
# import os
# import threading
# import subprocess

# app = Flask(__name__)
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file part'})
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({'error': 'No selected file'})
    
#     file_path = os.path.join(UPLOAD_FOLDER, file.filename)
#     file.save(file_path)
#     return jsonify({'success': True, 'filename': file.filename, 'filepath': file_path})

# @app.route('/process', methods=['POST'])
# def process_files():
#     data = request.json
#     text_file = data.get('text_file')
#     speaker_wav = data.get('speaker_wav')
#     face_video = data.get('face_video')
#     output_video = os.path.join(UPLOAD_FOLDER, "output.mp4")
    
#     if not all([text_file, speaker_wav, face_video]):
#         return jsonify({'error': 'Missing input files'})
    
#     def process():
#         output_wav = os.path.join(UPLOAD_FOLDER, "output.wav")
#         subprocess.run(["python3", "tts_script.py", text_file, speaker_wav, output_wav])
#         subprocess.run(["python3", "wav2lip_script.py", face_video, output_wav, output_video])

#     thread = threading.Thread(target=process)
#     thread.start()
    
#     return jsonify({'success': True, 'message': 'Processing started'})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
import subprocess
import os

app = Flask(__name__)

@app.route('/lip-sync', methods=['POST'])
def lip_sync():
    try:
        data = request.json
        text_path = data.get('text', '../input_script_0.txt')
        audio_path = data.get('audio', '../inputs/audio_input_files/combined_samples.wav')
        face_path = data.get('face', '../inputs/video_input_files/hey_gen_2.mp4')
        output_audio = data.get('output_audio', '../outputs/combined_speaker_script0/00_XTTS_v2_combined_speaker_script_0.wav')
        output_video = data.get('output_video', 'output.mp4')

        if not all([text_path, audio_path, face_path]):
            return jsonify({"error": "Missing required inputs"}), 400

        command = [
            "python", "generate_lip_sync.py",
            "--text", text_path,
            "--audio", audio_path,
            "--face", face_path,
            "--output_audio", output_audio,
            "--output_video", output_video
        ]
        
        subprocess.run(command, check=True)
        
        return jsonify({"message": "Lip-sync process completed", "output_video": output_video})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
