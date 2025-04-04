import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Function to load the criminal database
def load_database(filename="criminal_db.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}

# Function to train the KNN classifier
def train_knn_classifier(embeddings, labels):
    # Encode the labels (criminal names) to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Train the KNN model
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(embeddings, labels_encoded)

    # Save the trained model and label encoder
    with open("knn_model.pkl", "wb") as model_file:
        pickle.dump(knn, model_file)

    with open("label_encoder.pkl", "wb") as label_file:
        pickle.dump(label_encoder, label_file)

    print("KNN model trained and saved.")

# Load the criminal database
criminal_db = load_database()

# Extract embeddings and labels (criminal names)
embeddings = []
labels = []

for name, details in criminal_db.items():
    embeddings.append(details["embedding"])
    labels.append(name)

# Convert to numpy arrays for further processing
embeddings = np.array(embeddings)
labels = np.array(labels)

print(f"Loaded {len(embeddings)} criminals.")

# Train the KNN model
train_knn_classifier(embeddings, labels)
