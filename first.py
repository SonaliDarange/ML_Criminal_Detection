import os
import numpy as np
import pickle
from deepface import DeepFace

# Function to extract face embedding from an image
def get_face_embedding(image_path):
    try:
        embedding = DeepFace.represent(image_path, model_name="Facenet")[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"Error extracting embedding from image {image_path}: {e}")
        return None

# Function to load the criminal database
def load_database(filename="criminal_db.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

# Function to save the updated criminal database
def save_database(database, filename="criminal_db.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(database, f)

# Manually add criminal details and extract embeddings from their images
def add_criminals():
    criminal_db = load_database()

    # Define your criminals and image paths
    criminals = [
        {"name": "Reffeala", "age": 30, "crime": "Robbery", "image_path": "C:\\Users\\user\\Documents\\Downloads\\criminal2data.jpg"},
          {"name": "Merrian ", "age": 20, "crime": "froud", "image_path": "C:\\Users\\user\\Documents\\Downloads\\criminal3data.jpg"},
          {"name": "Asfak", "age": 40, "crime": "murder", "image_path": "C:\\Users\\user\\Documents\\Downloads\\criminal4data.jpg"},
          {"name": "samina", "age": 37, "crime": "half murder", "image_path": "C:\\Users\\user\\Documents\\Downloads\\criminal1data.jpeg"},
    
        # Add more criminals and their images here
    ]
    
    for criminal in criminals:
        name = criminal["name"]
        age = criminal["age"]
        crime = criminal["crime"]
        image_path = criminal["image_path"]

        # Extract face embedding for each criminal's image
        embedding = get_face_embedding(image_path)
        
        if embedding is not None:
            # Store criminal details and embedding in the database
            criminal_db[name] = {
                "embedding": embedding,
                "age": age,
                "crime": crime
            }
            print(f"✅ {name} added successfully.")
        else:
            print(f"❌ Failed to extract embedding for {name}.")
    
    # Save the updated criminal database to Pickle
    save_database(criminal_db)
    print("Criminal database saved.")

# Add criminals
add_criminals()
