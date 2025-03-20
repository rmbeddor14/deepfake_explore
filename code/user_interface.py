# from flask import Flask, render_template, request, jsonify
# import os
# import subprocess

# app = Flask(__name__)
# UPLOAD_FOLDER = "uploads"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     text_file = request.files.get('text')
#     audio_file = request.files.get('audio')
#     face_file = request.files.get('face')
    
#     if not all([text_file, audio_file, face_file]):
#         return jsonify({'error': 'All files must be provided'}), 400
    
#     text_path = os.path.join(UPLOAD_FOLDER, text_file.filename)
#     audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
#     face_path = os.path.join(UPLOAD_FOLDER, face_file.filename)
    
#     text_file.save(text_path)
#     audio_file.save(audio_path)
#     face_file.save(face_path)
    
#     output_audio = os.path.join(UPLOAD_FOLDER, "output_audio.wav")
#     output_video = os.path.join(UPLOAD_FOLDER, "output.mp4")
    
#     try:
#         command = [
#             "python3", "generate_lip_sync.py",
#             "--text", text_path,
#             "--audio", audio_path,
#             "--face", face_path,
#             "--output_audio", output_audio,
#             "--output_video", output_video
#         ]
#         subprocess.run(command, check=True)
#         return jsonify({'message': 'Processing complete', 'output_video': output_video})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, render_template, request, jsonify
import os
import subprocess
import threading
import shutil  # Added for deleting the folder

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lock to ensure only one process runs at a time
processing_lock = threading.Lock()

# def clear_upload_folder():
#     """Deletes all files in the upload folder before a new process starts."""
#     if os.path.exists(UPLOAD_FOLDER):
#         shutil.rmtree(UPLOAD_FOLDER)  # Delete the entire folder
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Recreate an empty folder

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def check_status():
    return jsonify({'processing': processing_lock.locked()})

@app.route('/upload', methods=['POST'])
def upload_files():
    if processing_lock.locked():
        return jsonify({'error': 'Another video is currently being processed. Please wait.'}), 429

    with processing_lock:  # Ensures only one process at a time
      #  clear_upload_folder()  # Clears previous uploads

        text_file = request.files.get('text')
        audio_file = request.files.get('audio')
        face_file = request.files.get('face')

        if not all([text_file, audio_file, face_file]):
            return jsonify({'error': 'All files must be provided'}), 400

        text_path = os.path.join(UPLOAD_FOLDER, text_file.filename)
        audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        face_path = os.path.join(UPLOAD_FOLDER, face_file.filename)

        text_file.save(text_path)
        audio_file.save(audio_path)
        face_file.save(face_path)

        output_audio = os.path.join(UPLOAD_FOLDER, "output_audio.wav")
        output_video = os.path.join(UPLOAD_FOLDER, "output.mp4")

        try:
            command = [
                "python3", "generate_lip_sync.py",
                "--text", text_path,
                "--audio", audio_path,
                "--face", face_path,
                "--output_audio", output_audio,
                "--output_video", output_video
            ]
            subprocess.run(command, check=True)
            return jsonify({'message': 'Processing complete', 'output_video': output_video})
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
