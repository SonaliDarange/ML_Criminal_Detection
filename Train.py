import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def load_database(filename="criminal_db.pkl"):
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {}


def train_knn_classifier(embeddings, labels):
   
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

   
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(embeddings, labels_encoded)

    
    with open("knn_model.pkl", "wb") as model_file:
        pickle.dump(knn, model_file)

    with open("label_encoder.pkl", "wb") as label_file:
        pickle.dump(label_encoder, label_file)

    print("KNN model trained and saved.")


criminal_db = load_database()


embeddings = []
labels = []

for name, details in criminal_db.items():
    embeddings.append(details["embedding"])
    labels.append(name)


embeddings = np.array(embeddings)
labels = np.array(labels)

print(f"Loaded {len(embeddings)} criminals.")


train_knn_classifier(embeddings, labels)
