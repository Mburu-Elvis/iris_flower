import numpy as np
from flask import Flask, request, jsonify, render_template, Response
import pickle
from sklearn.preprocessing import StandardScaler
# Create a Flask instance
app = Flask(__name__)

# load the pickle model
with open('model.pkl', 'rb') as file:
    model, scaler = pickle.load(file)

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        sepal_length = float(request.form.get('Sepal_Length'))
        sepal_width = float(request.form.get('Sepal_Width'))
        petal_length = float(request.form.get('Petal_Length'))
        petal_width = float(request.form.get('Petal_Width'))
        
        # Perform your prediction here using the input values
        arr = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        scaled_data = scaler.transform(arr)
        predictions = model.predict(scaled_data)
        print(f"Prediction Result: {predictions}")
        response =  jsonify({"Prediction": predictions.tolist()})
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return render_template('index.html', prediction_text=predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)