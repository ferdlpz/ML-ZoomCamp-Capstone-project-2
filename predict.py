import pickle

from flask import Flask, request, jsonify
import pandas as pd


model_file = 'model_max_depth=10_n_estimators=260.bin'

import os
os.chdir('/Users/fdl/Repos/ML-ZoomCamp-Capstone-project-2/')

with open(model_file, 'rb') as f_in:
    model, ct = pickle.load(f_in)

app = Flask('exam-predict')

@app.route('/predict', methods=['POST'])
def predict():
    student = request.get_json()
    df_student = pd.DataFrame([student])
    X = ct.transform(df_student)
    prediction = model.predict(X)[0]
    
    result = {
        'exam_score_prediction': float(round(prediction, 2))
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)