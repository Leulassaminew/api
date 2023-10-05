from flask import Flask, request , jsonify
import json
from urllib.parse import unquote
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)

columns = ['pop','city','lat','capacity','container','price','brand']

with open("model.pkl","rb") as f:
    model_body = pickle.load(f)
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
@app.route('/'):
def pri():
    return "home"
@app.route('/predict',methods=['POST'])
def predict_Sales():
    data=request.get_json()
    my_list = data["list"]
    new_input=np.array([my_list])
    c= scaler.transform(new_input)
    reshaped_input = c.reshape(1, -1)
    res = model_body.predict(reshaped_input)
    response = {'code':200,'status':'OK',
                'result':str(res[0])}
    return jsonify(response)

if __name__ == "__main__":
    app.run()
