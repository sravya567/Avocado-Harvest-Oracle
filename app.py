import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='template')

model = pickle.load(open(r'xgb_avocado.pkl', 'rb'))

@app.route('/')  # rendering the html template
def home():
    return render_template('index.html')

@app.route('/inner-page')  # rendering the html template
def output():
    return render_template("inner-page.html")

@app.route('/submit', methods=["POST"])  # route to show the predictions in a web UI
@app.route('/submit', methods=["POST"])
def submit():
    input_feature = [float(x) for x in request.form.values() if x]
    input_feature = [np.array(input_feature)]
    
    names = ['small/medium Avocados', 'large Avocados', 'extra_large Avocados', 'Small Bags', 'Large Bags', 'XLarge Bags', 'year', 'Month', 'Day', 'type_encoded', 'region_encoded']
    data = pd.DataFrame(input_feature, columns=names)

    prediction = model.predict(data)
    out = prediction[0]

    return render_template("output.html", result=out)


# running the app
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=7777, debug=False)
