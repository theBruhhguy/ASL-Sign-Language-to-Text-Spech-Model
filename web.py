from flask import Flask, render_template, request
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Initialize Flask application
app = Flask(__name__)

# Load your trained ML model
model = tf.keras.models.load_model(r'C:\Users\allen\Desktop\VIT\Semester 6\Projects\Hand Signs Model Project\In Progress\asl_model.h5')

# Define a function to preprocess the image before passing it to the model
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(your_image_size))  # Define your image size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle the form submission
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the file from the POST request
        file = request.files['file']
        # Save the file to the uploads folder
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        # Preprocess the image
        processed_image = preprocess_image(file_path)
        # Make prediction
        prediction = model.predict(processed_image)
        # Convert prediction to text or label
        # You need to implement this based on how your model is trained
        # For example, if your model outputs probabilities, you might need to map them to labels
        predicted_label = "Some code to get the predicted label"
        return predicted_label

if __name__ == '__main__':
    app.run(debug=True)
