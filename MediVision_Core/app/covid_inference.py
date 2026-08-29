import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the saved model
model = load_model('covid_classifier_mobilenet.keras')

# Function to preprocess the image
def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# Function to predict the class of an image
def predict_image_class(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    probability = prediction[0][0]
    if probability > 0.5:
        return 'COVID-19', probability
    else:
        return 'Normal', 1 - probability