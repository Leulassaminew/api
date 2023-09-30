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
@app.route('/api',methods=['GET'])
def returnascii():
    d={}
    #inputchar = list(request.args['query'])
    #answer = inputchar
    json_data = unquote(request.args.get('query'))
    decoded_data = json.loads(json_data)
    my_list = decoded_data
    new_input=np.array([my_list])
    c= scaler.transform(new_input)
    reshaped_input = c.reshape(1, -1)
    res = model_body.predict(reshaped_input)
    response = {'code':200,'status':'OK',
                'result':str(res[0])}
    return jsonify(response)

if __name__ == "__main__":
    app.run()