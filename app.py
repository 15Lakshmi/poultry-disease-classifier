from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model("poultry_disease_model.h5")

# Define class labels (must match training order)
classes = ['Coccidiosis', 'Healthy', 'NewCastle', 'Salmonella']

# Folder to store uploaded images
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('predict.html', prediction=None)

        file = request.files['image']

        if file.filename == '':
            return render_template('predict.html', prediction=None)

        # Save uploaded file to static/uploads
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Load and preprocess the image
        img = load_img(filepath, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = classes[np.argmax(prediction)]

        # Render prediction page with result
        return render_template('predict.html', prediction=predicted_class, image_path=filepath)

    # If method is GET, show page without prediction
    return render_template('predict.html', prediction=None)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
