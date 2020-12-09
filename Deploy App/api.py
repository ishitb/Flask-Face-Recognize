# Dependencies
from flask import Flask, request, jsonify,make_response
import pickle
import traceback
import numpy as np
import urllib.request
import sys
from recog import FaceRecognize

model = pickle.load(open('hello.pkl','rb'))
# Your API definition
app = Flask(__name__)

# @app.route('/')
# def man():
#     return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():   
    req = request.get_json()
    filename = 'face.jpg'
    #image_url = "https://thumbs.dreamstime.com/z/happy-little-boy-smiley-face-portrait-human-concept-freshness-133726078.jpg"
    image_url = str(req['URL'])
    urllib.request.urlretrieve(image_url,filename)
    print(req)
    pred = model.predict(filename,req['ROLLNO'])
    check = {"message" :str(pred)}
    print(check)
    return check        


if __name__ == '__main__':
    app.run(debug=True)

# app.run(debug=True)