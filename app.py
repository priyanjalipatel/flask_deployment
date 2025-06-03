import pickle
import numpy as np
from flask import Flask, request, render_template

# Step 1: Initialize the Flask app
app = Flask(__name__)

# Step 2: Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Step 3: Define the homepage route
@app.route('/')
def home():
    return render_template('index.html')  # Loads HTML form

# Step 4: Define the route for prediction
@app.route('/predict', methods=['POST'])  # ✅ Fixed parenthesis
def predict():
    # Step 4a: Get input from HTML form
    init_features = float(request.form['time'])  # ✅ 'time' is the name attribute in <input>

    # Step 4b: Convert input to NumPy array for prediction
    y_array = np.asarray(init_features)
    final_features = y_array.reshape(-1, 1)  # 2D array expected by model

    # Step 4c: Make prediction
    prediction = model.predict(final_features)

    # Step 4d: Show prediction result on same page
    return render_template('index.html', prediction_text='Predicted Class: {}'.format(prediction[0]))

# Step 5: Run the Flask app (for development/testing)
if __name__ == "__main__":  # ✅ Fixed semicolon
    app.run(debug=True)

