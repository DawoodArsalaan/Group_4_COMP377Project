from flask import Flask, jsonify, request
from model import fetch_and_train_model, predict_prices

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    coin = data.get('coin', 'bitcoin')
    days = data.get('days', 30)

    # Fetch data and train model
    try:
        predictions = fetch_and_train_model(coin, days)
        return jsonify({"predictions": predictions}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
