from flask import Flask, request, jsonify
from analyse import analyse_tos, SentenceTransformerFeatures
from flask_cors import CORS

app = Flask(__name__)
CORS(app)    

@app.route('/analyse', methods=['POST'])
def analyse():
    # Get the value from the post request
    print("Received request")
    data = request.get_json()
    tos = data.get('tos')
    appName = data.get('appName')
    categorized_sentences = analyse_tos(tos, appName)
    return jsonify({"danger": categorized_sentences[2],
                    "warning": categorized_sentences[1],
                    "normal": categorized_sentences[0],
                    "danger_summary": categorized_sentences[5],
                    "warning_summary": categorized_sentences[4],
                    "normal_summary": categorized_sentences[3]})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)