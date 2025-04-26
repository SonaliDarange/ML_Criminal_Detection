import os
import numpy as np
import pickle
import cv2
from deepface import DeepFace
from flask import Flask, request, render_template, Response, redirect, url_for
from threading import Lock

app = Flask(__name__, template_folder="C:\\ML\\templates", static_folder="C:\\ML\\static")

UPLOAD_FOLDER = "C:\\ML\\static\\uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DATABASE_FILE = "criminal_db.pkl"


last_match_info = {"matched": False, "details": None}
lock = Lock()


def load_database():
    try:
        with open(DATABASE_FILE, "rb") as f:
            database = pickle.load(f)
            print("[INFO] Criminal database loaded.")
            return database
    except FileNotFoundError:
        print("[WARNING] Database file not found.")
        return {}

def get_face_embedding(image_path):
    try:
        embeddings = DeepFace.represent(img_path=image_path, model_name="Facenet", enforce_detection=False)
        if embeddings:
            return np.array(embeddings[0]["embedding"])
        else:
            return None
    except Exception as e:
        print(f"[ERROR] Embedding failed: {e}")
        return None


def identify_criminal(image_path):
    database = load_database()
    query_embedding = get_face_embedding(image_path)

    if query_embedding is None:
        return "No face detected", None

    for name, details in database.items():
        db_embedding = details.get("embedding")
        if db_embedding is None:
            continue

        distance = np.linalg.norm(np.array(db_embedding) - query_embedding)
        if distance < 10:
            print(f"[MATCH] {name} | Distance: {distance}")
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
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    result, criminal_info = identify_criminal(path)
    return render_template(
        'index.html',
        result=result,
        criminal_info=criminal_info,
        image_url=f"/static/uploads/{filename}"
    )

@app.route('/start_surveillance')
def start_surveillance():
    with lock:
        last_match_info["matched"] = False
        last_match_info["details"] = None
    return render_template('surveillance.html')

def generate_frames():
    cap = cv2.VideoCapture(0)
    database = load_database()

    while True:
        success, frame = cap.read()
        if not success:
            break

        temp_path = os.path.join(UPLOAD_FOLDER, "frame.jpg")
        cv2.imwrite(temp_path, frame)

        embedding = get_face_embedding(temp_path)
        matched = False

        if embedding is not None:
            for name, details in database.items():
                db_embedding = details.get("embedding")
                if db_embedding is None:
                    continue

                distance = np.linalg.norm(np.array(db_embedding) - embedding)
                if distance < 10:
                    text = f"{name} | {details.get('crime', '')} | {details.get('status', '')}"
                    cv2.putText(frame, text, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    with lock:
                        last_match_info["matched"] = True
                        last_match_info["details"] = details
                        last_match_info["details"]["name"] = name
                    matched = True
                    break

        if not matched:
            cv2.putText(frame, "No match found", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_report')
def video_report():
    with lock:
        if last_match_info["matched"]:
            return render_template(
                'index.html',
                result="Match found from video surveillance",
                criminal_info=last_match_info["details"],
                image_url="/static/uploads/frame.jpg"
            )
        else:
            return render_template(
                'surveillance.html',
                result="No match found yet. Please wait..."
            )

if __name__ == "__main__":
    app.run(debug=True)
