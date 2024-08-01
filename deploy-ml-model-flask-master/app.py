from flask import Flask, render_template, request
from model import predict
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_class():
    features = {
        'age': request.form['age'],
        'menopause': request.form['menopause'],
        'tumor-size': request.form['tumor-size'],
        'inv-nodes': request.form['inv-nodes'],
        'node-caps': request.form['node-caps'],
        'deg-malig': request.form['deg-malig'],
        'breast': request.form['breast'],
        'breast-quad': request.form['breast-quad'],
        'irradiat': request.form['irradiat']
    }

    features_df = pd.DataFrame([features])
    prediction = predict(features_df)

    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
