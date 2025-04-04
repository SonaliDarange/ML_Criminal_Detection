import pickle
import numpy as np

# Function to load the criminal database
def load_database(filename="criminal_db.pkl"):
    try:
        with open(filename, "rb") as f:
            database = pickle.load(f)
            print("Database Loaded Successfully:", database)  # Debugging
            return database
    except FileNotFoundError:
        print("Error: Database file not found!")
        return {}

# Load criminal database
criminal_db = load_database()

# Extract embeddings and labels (criminal names)
embeddings = []
labels = []

for name, details in criminal_db.items():
    embedding = details.get("embedding")  # Avoid KeyError
    if embedding is not None:  # Ensure embedding exists
        embeddings.append(embedding)
        labels.append(name)
    else:
        print(f"Warning: No embedding found for {name}")

# Convert to numpy arrays (if embeddings exist)
if embeddings:
    embeddings = np.array(embeddings)
    labels = np.array(labels)
    print(f"Loaded {len(embeddings)} criminals into the system.")
else:
    print("No valid embeddings found in the database.")
