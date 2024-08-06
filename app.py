from flask import Flask, render_template, request
import joblib
import json
import pandas as pd
from joblib import load

model = load('model.pkl')

with open('feature_columns.json', 'r') as f:
    loaded_feature_columns = json.load(f)

app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_price():
    area = float(request.form.get('Area'))
    bedrooms = int(request.form.get('No. of Bedrooms'))
    MaintenanceStaff = int(request.form.get('MaintenanceStaff'))
    Security = int(request.form.get('24X7Security'))
    Latitude = float(request.form.get('Latitude'))
    Longitude = float(request.form.get('Longitude'))

    new_input = [[area, bedrooms, MaintenanceStaff, Security, Latitude, Longitude]]

    new_input_df = pd.DataFrame(new_input, columns=loaded_feature_columns)

    prediction = model.predict(new_input_df)

    return render_template('index.html', result=str(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)