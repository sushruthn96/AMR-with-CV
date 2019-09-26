from flask import Flask, request, jsonify
from predict import Predict
import cv2

app = Flask(__name__)

# Specify the path for your model file
model_file = '../models/epochs10_demo.hd5'

# Specify the path for your csv file which contains the mappings for ids vs classes
csv_file = model_file.replace('hd5', 'csv')

# Create Predict object
predict_obj = Predict.Predict(model_file, csv_file)


@app.route('/predict', methods=['POST'])
def get_prediction():
    if request.method == 'POST':
        if 'image' in request.files:
            print("hello")
            file = request.files['image']
            filename = "some_filename.jpg"
            file.save(filename)
            cv2_image = cv2.imread(filename)
            results = predict_obj.get_prediction(cv2_image)
            return jsonify(results)
        return "Bad request"
    else:
        return "405: Undefined"


@app.route('/sample')
def sample():
    return "Hello! I am up!"


app.run("0.0.0.0", port=5000)
