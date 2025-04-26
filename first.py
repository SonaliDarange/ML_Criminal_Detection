import os
import numpy as np
import pickle
from deepface import DeepFace


def get_face_embedding(image_path):
    try:
        embedding = DeepFace.represent(image_path, model_name="Facenet")[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"Error extracting embedding from image {image_path}: {e}")
        return None


def load_database(filename="criminal_db.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def save_database(database, filename="criminal_db.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(database, f)

def add_criminals():
    criminal_db = load_database()

    criminals = [
        {
            "name": "Reffeala",
            "alias": "Red Fox",
            "age": 30,
            "gender": "Female",
            "crime": "Robbery",
            "crime_date": "2023-06-10",
            "location": "Delhi",
            "status": "Wanted",
            "case_number": "DLR2023-001",
            "image_path": "C:\\Users\\user\\Documents\\Downloads\\criminal2data.jpg"
        },
        {
            "name": "Merrian",
            "alias": "Mimi",
            "age": 20,
            "gender": "Female",
            "crime": "Fraud",
            "crime_date": "2024-01-03",
            "location": "Mumbai",
            "status": "Convicted",
            "case_number": "MUM2024-002",
            "image_path": "C:\\Users\\user\\Documents\\Downloads\\criminal3data.jpg"
        },
        {
            "name": "Asfak",
            "alias": "AK47",
            "age": 40,
            "gender": "Male",
            "crime": "Murder",
            "crime_date": "2022-05-05",
            "location": "Kolkata",
            "status": "Wanted",
            "case_number": "KOL2022-007",
            "image_path": "C:\\Users\\user\\Documents\\Downloads\\criminal4data.jpg"
        },
        {
            "name": "Samina",
            "alias": "Queen Cobra",
            "age": 37,
            "gender": "Female",
            "crime": "Attempted Murder",
            "crime_date": "2023-12-12",
            "location": "Bangalore",
            "status": "Under Investigation",
            "case_number": "BLR2023-009",
            "image_path": "C:\\Users\\user\\Documents\\Downloads\\criminal1data.jpeg"
        }, {
            "name": "sonali",
            "alias": "Q",
            "age": 20,
            "gender": "Female",
            "crime": "no ",
            "crime_date": "2023-12-12",
            "location": "Bangalore",
            "status": "Under Investigation",
            "case_number": "BLR2023-009",
            "image_path": "C:\\Users\\user\\Pictures\\scholarship\\Passportphoto.jpg"
        }
    ]

    for criminal in criminals:
        name = criminal["name"]
        image_path = criminal["image_path"]

        embedding = get_face_embedding(image_path)
        if embedding is not None:
           
            criminal_info = {
                "name": criminal["name"],
                "alias": criminal["alias"],
                "age": criminal["age"],
                "gender": criminal["gender"],
                "crime": criminal["crime"],
                "crime_date": criminal["crime_date"],
                "location": criminal["location"],
                "status": criminal["status"],
                "case_number": criminal["case_number"],
                "embedding": embedding
            }

            criminal_db[name] = criminal_info
            print(f"✅ {name} added successfully.")
        else:
            print(f"❌ Failed to extract embedding for {name}.")

    save_database(criminal_db)
    print("✅ Criminal database saved with essential information.")


add_criminals()
