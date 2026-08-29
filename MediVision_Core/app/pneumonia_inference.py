import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = load_model('model_95.h5')

# Function to preprocess the image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(150, 150))  # Adjust target size to match model input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to predict the class of an image
def predict_image_class(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array).ravel()
    probability = prediction[0]
    prediction_class = (probability > 0.5).astype(int)
    class_names = {0: "Normal", 1: "Pneumonia"}
    
    if prediction_class == 0:
        probability = 1 - probability
    
    return class_names[prediction_class], probability