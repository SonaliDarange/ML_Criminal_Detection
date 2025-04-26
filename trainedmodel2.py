import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


def load_database(filename="criminal_db.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print("❌ Criminal database not found. Run add_criminals() first.")
        return {}


def train_knn_classifier():
    criminal_db = load_database()
    
    if not criminal_db:
        print("❌ No data found. Add criminals before training.")
        return None, None

    embeddings = []
    labels = []

    for name, details in criminal_db.items():
        embeddings.append(details["embedding"])
        labels.append(name)

    
    embeddings = np.array(embeddings)
    labels = np.array(labels)

    
    knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
    knn.fit(embeddings, labels)

   
    with open("trained_model.pkl", "wb") as f:
        pickle.dump(knn, f)

    print("✅ Model trained successfully and saved as trained_model.pkl")

train_knn_classifier()
