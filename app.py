from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import cv2
import numpy as np
import os
import io
import uuid
from datetime import datetime
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB max upload

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


# Helper function to process uploaded files
def process_uploaded_file(file, filename):
    """Save uploaded file and return its path"""
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    return file_path


# Show Image
@app.route('/show_image', methods=['POST'])
def show_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    file_path = process_uploaded_file(file, filename)

    return send_file(file_path, mimetype='image/*')


# Image Blending
@app.route('/blend_images', methods=['POST'])
def blend_images():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Please upload both images"}), 400

    # Process both images
    img1 = request.files['image1']
    img2 = request.files['image2']

    # Read images as numpy arrays
    img1_np = np.frombuffer(img1.read(), np.uint8)
    img2_np = np.frombuffer(img2.read(), np.uint8)

    img1_cv = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
    img2_cv = cv2.imdecode(img2_np, cv2.IMREAD_COLOR)

    # Resize images to match
    height = min(img1_cv.shape[0], img2_cv.shape[0])
    width = min(img1_cv.shape[1], img2_cv.shape[1])
    img1_resized = cv2.resize(img1_cv, (width, height))
    img2_resized = cv2.resize(img2_cv, (width, height))

    # Blend images
    blended = cv2.addWeighted(img1_resized, 0.5, img2_resized, 0.5, 0)

    # Convert to bytes
    _, img_encoded = cv2.imencode('.png', blended)
    return send_file(
        io.BytesIO(img_encoded),
        mimetype='image/png',
        download_name='blended.png'
    )


# Reshape Image
@app.route('/reshape_image', methods=['POST'])
def reshape_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    width = int(request.form.get('width', 300))
    height = int(request.form.get('height', 300))

    # Read image
    img_np = np.frombuffer(file.read(), np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Resize image
    resized = cv2.resize(img_cv, (width, height))

    # Convert to bytes
    _, img_encoded = cv2.imencode('.png', resized)
    return send_file(
        io.BytesIO(img_encoded),
        mimetype='image/png',
        download_name='reshaped.png'
    )


# Weighted Blending
@app.route('/weighted_blend', methods=['POST'])
def weighted_blend():
    if 'image1' not in request.files or 'image2' not in request.files:
        return jsonify({"error": "Please upload both images"}), 400

    img1 = request.files['image1']
    img2 = request.files['image2']
    weight = float(request.form.get('weight', 0.5))

    # Read images
    img1_np = np.frombuffer(img1.read(), np.uint8)
    img2_np = np.frombuffer(img2.read(), np.uint8)

    img1_cv = cv2.imdecode(img1_np, cv2.IMREAD_COLOR)
    img2_cv = cv2.imdecode(img2_np, cv2.IMREAD_COLOR)

    # Resize images to match
    height = min(img1_cv.shape[0], img2_cv.shape[0])
    width = min(img1_cv.shape[1], img2_cv.shape[1])
    img1_resized = cv2.resize(img1_cv, (width, height))
    img2_resized = cv2.resize(img2_cv, (width, height))

    # Apply weighted blending
    blended = cv2.addWeighted(img1_resized, weight, img2_resized, 1 - weight, 0)

    # Convert to bytes
    _, img_encoded = cv2.imencode('.png', blended)
    return send_file(
        io.BytesIO(img_encoded),
        mimetype='image/png',
        download_name='weighted_blend.png'
    )


# ROI Extraction
@app.route('/extract_roi', methods=['POST'])
def extract_roi():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    x = int(request.form.get('x', 0))
    y = int(request.form.get('y', 0))
    width = int(request.form.get('width', 100))
    height = int(request.form.get('height', 100))

    # Read image
    img_np = np.frombuffer(file.read(), np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Extract ROI
    roi = img_cv[y:y + height, x:x + width]

    # Convert to bytes
    _, img_encoded = cv2.imencode('.png', roi)
    return send_file(
        io.BytesIO(img_encoded),
        mimetype='image/png',
        download_name='roi.png'
    )


# Geometric Shapes
@app.route('/add_shape', methods=['POST'])
def add_shape():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    shape_type = request.form.get('shape', 'rectangle')
    color = request.form.get('color', '#00bcd4').lstrip('#')
    color = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
    thickness = int(request.form.get('thickness', 2))
    x1 = int(request.form.get('x1', 50))
    y1 = int(request.form.get('y1', 50))
    x2 = int(request.form.get('x2', 150))
    y2 = int(request.form.get('y2', 150))

    # Read image
    img_np = np.frombuffer(file.read(), np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Draw shapes
    if shape_type == 'line':
        cv2.line(img_cv, (x1, y1), (x2, y2), color, thickness)
    elif shape_type == 'arrowed_line':
        cv2.arrowedLine(img_cv, (x1, y1), (x2, y2), color, thickness)
    elif shape_type == 'rectangle':
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, thickness)
    elif shape_type == 'circle':
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        radius = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) // 2)
        cv2.circle(img_cv, center, radius, color, thickness)

    # Convert to bytes
    _, img_encoded = cv2.imencode('.png', img_cv)
    return send_file(
        io.BytesIO(img_encoded),
        mimetype='image/png',
        download_name='shape_image.png'
    )


# Add Text to Image
@app.route('/add_text', methods=['POST'])
def add_text():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    text = request.form.get('text', 'Sample Text')
    x = int(request.form.get('x', 50))
    y = int(request.form.get('y', 50))
    size = float(request.form.get('size', 30))
    color = request.form.get('color', '#ffffff').lstrip('#')
    color = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
    thickness = int(request.form.get('thickness', 2))

    # Read image
    img_np = np.frombuffer(file.read(), np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Add text
    cv2.putText(img_cv, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

    # Convert to bytes
    _, img_encoded = cv2.imencode('.png', img_cv)
    return send_file(
        io.BytesIO(img_encoded),
        mimetype='image/png',
        download_name='text_image.png'
    )


# Add Date/Time to Image
@app.route('/add_datetime', methods=['POST'])
def add_datetime():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    format_type = request.form.get('format', 'datetime')
    x = int(request.form.get('x', 50))
    y = int(request.form.get('y', 50))
    size = float(request.form.get('size', 30))
    color = request.form.get('color', '#ffffff').lstrip('#')
    color = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
    thickness = int(request.form.get('thickness', 2))

    # Get current date/time
    now = datetime.now()
    if format_type == 'datetime':
        dt_str = now.strftime("%Y-%m-%d %H:%M:%S")
    elif format_type == 'date':
        dt_str = now.strftime("%Y-%m-%d")
    elif format_type == 'time':
        dt_str = now.strftime("%H:%M:%S")
    else:  # custom
        custom_format = request.form.get('custom_format', '%Y-%m-%d %H:%M:%S')
        dt_str = now.strftime(custom_format)

    # Read image
    img_np = np.frombuffer(file.read(), np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Add text
    cv2.putText(img_cv, dt_str, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)

    # Convert to bytes
    _, img_encoded = cv2.imencode('.png', img_cv)
    return send_file(
        io.BytesIO(img_encoded),
        mimetype='image/png',
        download_name='datetime_image.png'
    )


# Face Detection (Image)
@app.route('/detect_faces', methods=['POST'])
def detect_faces():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    color = request.form.get('color', '#00bcd4').lstrip('#')
    color = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
    thickness = int(request.form.get('thickness', 3))

    # Read image
    img_np = np.frombuffer(file.read(), np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), color, thickness)

    # Add count text
    cv2.putText(
        img_cv,
        f"{len(faces)} faces detected",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2
    )

    # Convert to bytes
    _, img_encoded = cv2.imencode('.png', img_cv)
    return send_file(
        io.BytesIO(img_encoded),
        mimetype='image/png',
        download_name='face_detection.png'
    )


# Face Detection (Video)
@app.route('/detect_faces_video', methods=['POST'])
def detect_faces_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    file = request.files['video']
    color = request.form.get('color', '#00bcd4').lstrip('#')
    color = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
    thickness = int(request.form.get('thickness', 3))
    show_datetime = request.form.get('show_datetime', 'false') == 'true'

    # Save video to temp file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    file.save(temp_video.name)
    temp_video.close()

    # Open video file
    cap = cv2.VideoCapture(temp_video.name)
    if not cap.isOpened():
        os.unlink(temp_video.name)
        return jsonify({"error": "Could not open video"}), 400

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output video path
    output_filename = f"face_detection_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Load face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        # Add face count text
        cv2.putText(
            frame,
            f"{len(faces)} faces",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        # Add datetime if requested
        if show_datetime:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                frame,
                now,
                (10, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # Write frame to output
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    os.unlink(temp_video.name)

    return send_file(output_path, mimetype='video/mp4')


# Edge Detection
@app.route('/detect_edges', methods=['POST'])
def detect_edges():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    threshold1 = int(request.form.get('threshold1', 100))
    threshold2 = int(request.form.get('threshold2', 200))
    edge_type = request.form.get('edge_type', 'canny')

    # Read image
    img_np = np.frombuffer(file.read(), np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Apply edge detection
    if edge_type == 'canny':
        edges = cv2.Canny(gray, threshold1, threshold2)
    elif edge_type == 'sobel':
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edges = cv2.magnitude(sobelx, sobely)
    elif edge_type == 'laplacian':
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        edges = np.uint8(np.absolute(laplacian))

    # Convert to color for output
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Convert to bytes
    _, img_encoded = cv2.imencode('.png', edges_color)
    return send_file(
        io.BytesIO(img_encoded),
        mimetype='image/png',
        download_name='edges.png'
    )


# Corner Detection
@app.route('/detect_corners', methods=['POST'])
def detect_corners():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    threshold = float(request.form.get('threshold', 0.01))

    # Read image
    img_np = np.frombuffer(file.read(), np.uint8)
    img_cv = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Detect corners
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    corners = cv2.dilate(corners, None)

    # Mark corners
    img_cv[corners > threshold * corners.max()] = [0, 0, 255]  # Red corners

    # Convert to bytes
    _, img_encoded = cv2.imencode('.png', img_cv)
    return send_file(
        io.BytesIO(img_encoded),
        mimetype='image/png',
        download_name='corners.png'
    )


if __name__ == '__main__':
    app.run(debug=True)
