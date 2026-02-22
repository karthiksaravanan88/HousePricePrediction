from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

regmodel = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    
    # Ensure features are in the correct order used during training
    feature_order = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat']
    feature_values = np.array([data[feature] for feature in feature_order]).reshape(1, -1)
    
    print("Input features:", feature_values)
    new_data = scaler.transform(feature_values)
    print("Scaled features:", new_data)
    
    output = regmodel.predict(new_data)
    print("Predicted price:", output[0])

    return jsonify(output[0])

if __name__ == "__main__":
    app.run(debug=True)