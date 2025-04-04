import os
import numpy as np
import pickle
import cv2
from deepface import DeepFace
from flask import Flask, request, render_template, Response

app = Flask(__name__, template_folder="C:\\ML\\templates", static_folder="C:\\ML\\static")

UPLOAD_FOLDER = "C:\\ML\\static\\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATABASE_FILE = "criminal_db.pkl"

def load_database():
    """ Load the criminal database from a pickle file. """
    try:
        with open(DATABASE_FILE, "rb") as f:
            database = pickle.load(f)
            print("Database loaded successfully:", database)
            return database
    except FileNotFoundError:
        print("Database file not found. Creating an empty database.")
        return {}

def get_face_embedding(image_path):
    """ Extract facial embedding from an image. """
    try:
        embeddings = DeepFace.represent(image_path, model_name="Facenet", enforce_detection=False)
        if embeddings:
            return np.array(embeddings[0]["embedding"])
        else:
            print(f"No embedding found for {image_path}")
            return None
    except Exception as e:
        print(f"Error extracting embedding from {image_path}: {e}")
        return None

def identify_criminal(image_path):
    """ Match uploaded image against the criminal database. """
    database = load_database()
    image_embedding = get_face_embedding(image_path)

    if image_embedding is None:
        return "No match found", None

    for name, details in database.items():
        print(f"Checking: {name}")
        stored_embedding = details.get("embedding")
        if stored_embedding is None:
            continue
        
        distance = np.linalg.norm(stored_embedding - image_embedding)
        if distance < 10:
            return f"Match found: {name}", details

    return "No match found", None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    result, criminal_info = identify_criminal(file_path)
    
    return render_template(
        'index.html',
        result=result,
        criminal_info=criminal_info,
        image_url=f"/static/uploads/{filename}"
    )

def generate_frames():
    """ Capture video frames, detect faces, and match against the database in real-time. """
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        temp_path = os.path.join(UPLOAD_FOLDER, "temp_frame.jpg")
        cv2.imwrite(temp_path, frame)
        result, criminal_info = identify_criminal(temp_path)

        if criminal_info:
            text = f"{criminal_info.get('name', 'Unknown')} - {criminal_info.get('crime', 'Unknown Crime')}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "No match found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    """ Stream live video to the webpage. """
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
