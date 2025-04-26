import pickle
import numpy as np


def load_database(filename="criminal_db.pkl"):
    try:
        with open(filename, "rb") as f:
            database = pickle.load(f)
            print("Database Loaded Successfully:", database) 
            return database
    except FileNotFoundError:
        print("Error: Database file not found!")
        return {}


criminal_db = load_database()


embeddings = []
labels = []

for name, details in criminal_db.items():
    embedding = details.get("embedding")  
    if embedding is not None: 
        embeddings.append(embedding)
        labels.append(name)
    else:
        print(f"Warning: No embedding found for {name}")


if embeddings:
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    print(f"Loaded {len(embeddings)} criminals into the system.")
else:
    print("No valid embeddings found in the database.")
