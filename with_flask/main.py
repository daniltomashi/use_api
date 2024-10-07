import sys
sys.path.insert(0, "/home/danylo/Desktop/New knowledge/use_api/")

from flask import Flask, render_template, request, jsonify
import functions_for_api
import pandas as pd



app = Flask(__name__, template_folder="templates")

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=["POST"])
def predict():
    # data = request.json
    data = request.form.to_dict(flat=False)

    result = functions_for_api.classify(data)
    return render_template('index.html', prediction=result)


if __name__ == "__main__":
    app.run(debug=True)