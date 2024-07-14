from flask import Flask, request, jsonify
from analyse import analyse_tos
from flask_cors import CORS
from ai import SentenceTransformerFeatures, POSTagFeatures, NERFeatures, KeywordFeatures, DependencyFeatures, SentimentFeatures
import pandas as pd
from os import path

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def index():
    return "Nice"

@app.route('/analyse', methods=['POST'])
def analyse():
    # Get the value from the post request
    print("Received request")
    data = request.get_json()
    tos = data.get('tos')
    appName = data.get('appName')
    url = data.get('url')
    categorized_sentences = analyse_tos(tos, appName, url)
    for i in range(len(categorized_sentences)):
        if type(categorized_sentences[i]) != str or not categorized_sentences[i]:
            categorized_sentences[i] = ""
    return jsonify({"danger": categorized_sentences[2],
                    "warning": categorized_sentences[1],
                    "normal": categorized_sentences[0],
                    "danger_summary": categorized_sentences[5],
                    "warning_summary": categorized_sentences[4],
                    "normal_summary": categorized_sentences[3]})

@app.route('/scans', methods=['GET'])
def scans():
    scans_path = path.join(path.dirname(__file__), '../scans.csv')
    scans = pd.read_csv(scans_path)
    scansList = list(scans['App'].values)
    print(scansList)
    return jsonify({"scans": scansList})

app.run(debug=True)