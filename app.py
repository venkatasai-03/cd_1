import os
from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import tensorflow as tf

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

categories = ['Mirchi', 'Mango', 'Lemon', 'Papaya', 'Potato', 'Tomato', 'Banana']
quality_labels = ['90', '70', '60']

IMG_SIZE = 128  # Image size used for training

# Load the model
model_path = 'final_model_1.h5'
model = tf.keras.models.load_model(model_path)

# Function to preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')  # Open image and convert to RGB
    image = image.resize((IMG_SIZE, IMG_SIZE))  # Resize to 128x128
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to make prediction
def predict_image(image_path):
    # Preprocess the image
    image_array = preprocess_image(image_path)

    # Make prediction
    predictions = model.predict(image_array)

    # Extract crop and quality predictions
    crop_pred = predictions[0]  # Crop type output
    quality_pred = predictions[1]  # Quality output

    # Get the indices of the maximum values (predicted class)
    crop_index = np.argmax(crop_pred)
    quality_index = np.argmax(quality_pred)

    # Map the indices to the actual labels
    predicted_crop = categories[crop_index]
    predicted_quality = quality_labels[quality_index]

    return predicted_crop, predicted_quality

# Route for the home page with upload form
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is submitted
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Save the uploaded file to static/uploads
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Run prediction
            predicted_crop, predicted_quality = predict_image(file_path)

            # Return the path relative to the 'static' folder
            image_url = url_for('static', filename='uploads/' + file.filename)

            return render_template('index.html', crop=predicted_crop, quality=predicted_quality, image_url=image_url)

    return render_template('index.html', crop=None, quality=None, image_url=None)

# Run the Flask app
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
