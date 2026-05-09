from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import cv2
import numpy as np
from gfpgan import GFPGANer
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load GFPGAN model once at startup
print("Loading GFPGAN model... This may take a moment.")
restorer = GFPGANer(
    model_path='C:/Users/ASUS/Downloads/GFPGANv1.4.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)
print("GFPGAN model loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')   # For initial page load

    # Handle POST request (from the modern UI)
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No selected file'}), 400

        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400

        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            input_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(input_path)

            # Read image
            img = cv2.imread(input_path)
            if img is None:
                return jsonify({'success': False, 'error': 'Could not read image'}), 400

            # Run GFPGAN restoration
            _, _, restored_img = restorer.enhance(
                img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )

            # Process and save restored image
            restored_img = np.clip(restored_img, 0, 255).astype(np.uint8)
            if len(restored_img.shape) == 4:
                restored_img = restored_img[0]

            output_filename = 'restored_' + filename
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            cv2.imwrite(output_path, restored_img)

            restored_url = f'/static/output/{output_filename}'

            return jsonify({
                'success': True,
                'restored_url': restored_url,
                'original_filename': filename
            })

        except Exception as e:
            print(f"Error during restoration: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)