from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

# Load the model
model = joblib.load("C:/Users/Aashir/Desktop/model/homeprice_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    area = data.get('area')
    if not isinstance(area, (int, float)):
        return jsonify({'error': 'Invalid input'}), 400
    price = model.predict(np.array([[area]]))
    return jsonify({'price': price[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

