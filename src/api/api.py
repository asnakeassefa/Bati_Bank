from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

# Load the saved model
model = joblib.load('../notebook/model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return "Fraud Detection Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_features = np.array(data['features']).reshape(1, -1)
        feature_names = [
            'Amount', 'Value', 'PricingStrategy',
            'TotalTransactionAmount', 'AverageTransactionAmount',
            'TransactionCount', 'TransactionAmountStd', 'TransactionHour',
            'TransactionDay', 'TransactionMonth', 'TransactionYear',
            'ProductCategory_data_bundles', 'ProductCategory_financial_services',
            'ProductCategory_movies', 'ProductCategory_other',
            'ProductCategory_ticket', 'ProductCategory_transport',
            'ProductCategory_tv', 'ProductCategory_utility_bill',
            'ChannelId_ChannelId_2', 'ChannelId_ChannelId_3',
            'ChannelId_ChannelId_5', 'Recency', 'Frequency', 'Monetary', 'Status',
            'RFMS_Score', 'Label'
        ]

        # Create a DataFrame with the input features
        input_df = pd.DataFrame(input_features, columns=feature_names)
        response = []
        # Make a prediction
        prediction = model.predict(input_df)
        for value in prediction:
            if value == 0:
                response.append(False)
            else:
                response.append(True)

        return jsonify({'prediction': response})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)
# Sample input for Postman test:
"""
    "features": [
        0.926715,
        0.000073,
        2,
        0.984208,
        0.88069,
        586,
        5268.094459,
        6,
        14,
        12,
        2018,
        0,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        0,
        0,
        0.0,
        0.142822,
        0.147025,
        0,
        -0.004203,
        1
    ]
"""