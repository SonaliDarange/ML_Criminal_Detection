from deepface import DeepFace
import cv2
import pickle
import numpy as np

# Function to load the saved KNN model and label encoder
def load_models():
    with open("knn_model.pkl", "rb") as model_file:
        knn_model = pickle.load(model_file)
    
    with open("label_encoder.pkl", "rb") as label_file:
        label_encoder = pickle.load(label_file)
    
    return knn_model, label_encoder

# Function to extract embedding from the image
def get_face_embedding(image_path):
    try:
        embedding = DeepFace.represent(image_path, model_name="Facenet")[0]["embedding"]
        return np.array(embedding)
    except Exception as e:
        print(f"Error extracting embedding from image {image_path}: {e}")
        return None

# Function to make a prediction based on the provided image
def predict_criminal(image_path, knn_model, label_encoder):
    # Get the embedding for the input image
    embedding = get_face_embedding(image_path)
    
    if embedding is not None:
        # Predict the closest match using the KNN model
        prediction = knn_model.predict([embedding])
        predicted_label = label_encoder.inverse_transform(prediction)
        
        return predicted_label[0]
    else:
        return None

# Load the trained model and label encoder
knn_model, label_encoder = load_models()

# Example: Predict criminal from a new image
image_path = "C:\\Users\\user\\Documents\\Downloads\\test_image.jpg"

predicted_criminal = predict_criminal(image_path, knn_model, label_encoder)

if predicted_criminal:
    print(f"Criminal identified: {predicted_criminal}")
else:
    print("No match found.")
