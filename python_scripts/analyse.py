import sys
import pickle
import os
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import os
from googlesearch import search
from bs4 import BeautifulSoup
import requests
import numpy as np
from scipy.sparse import csr_matrix
import warnings

# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

#Using torch
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

minilm_path = os.path.join(os.path.dirname(__file__), '../ai_models/minilm')
class SentenceTransformerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(minilm_path)
        self.model_name = model_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        embeddings = self.model.encode(X if type(X) == list else X.tolist(), convert_to_tensor=False)
        return np.array(embeddings)
    
def summarize(text):
    if text == '':
        return text
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    ouputs = model.generate(inputs, max_length=150, min_length=75, length_penalty=5.0, num_beams=2, early_stopping=True)
    return tokenizer.decode(ouputs[0], skip_special_tokens=True)

def predict(sentence):
    # Load the model from the file
    model_path = os.path.join(os.path.dirname(__file__), '../ai_models/model2.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model.predict([sentence])[0]

def analyse_tos(tos, app=""):
    scans_path = os.path.join(os.path.dirname(__file__), '../scans.csv')
    scans = pd.read_csv(scans_path)
    #online_scans_url = 'https://docs.google.com/spreadsheets/d/e/2PACX-1vQjd7DmxuwsQccfgX02enJf-g4DnWnvN5ZAkEHSfedfpqTF9JjYoSkvFUWNoTIy_PW6Kl_yhuzYtHy5/pub?gid=0&single=true&output=csv' 
    #online_scans = pd.read_csv(online_scans_url)
    if tos.strip()== '':
        print("No terms of service found for " + app + ". Searching the web...")
        tos_urls = search(app + " terms of service", num=1, stop=1)
        url = ''
        for i in tos_urls:
            url = i
        html_content = requests.get(url).text
        soup = BeautifulSoup(html_content, "html.parser")
        # Find all of the text between paragraph tags and strip out the html
        tos_list = soup.find_all('p')
        for i in tos_list:
            tos += i.get_text()
    print(scans['App'].values)
    if app in scans['App'].values:
        categorized_sentences = scans[scans['App'] == app].iloc[0].tolist()
        #print(categorized_sentences)
        categorized_sentences = scans[scans['App'] == app].iloc[0].tolist()[1:]
    #elif app in online_scans['App'].values:
    #    categorized_sentences = online_scans[online_scans['App'] == app].iloc[0].tolist()[1:]
    else:
        sentences = tos.split('.')
        categorized_sentences = [[], [], []]
        for sentence in sentences:
            categorized_sentences[predict(sentence)].append(sentence)
            #print(categorized_sentences[i])
        categorized_sentences = ["\n".join(categorized_sentences[0]), 
                                "\n".join(categorized_sentences[1]), 
                                "\n".join(categorized_sentences[2])]
        for i in range(3):
            categorized_sentences.append(summarize(categorized_sentences[i]))
        dct = {'App': app, 
                              'Level_0': categorized_sentences[0], 
                              'Level_1': categorized_sentences[1], 
                              'Level_2': categorized_sentences[2],
                              'Summary_0': categorized_sentences[3],
                              'Summary_1': categorized_sentences[4],
                              'Summary_2': categorized_sentences[5]}
        dct = {k:[v] for k,v in dct.items()}

        scans = pd.concat([scans, pd.DataFrame(dct)], 
                              ignore_index=True)
        scans.to_csv(scans_path, index=False)

    normal_path = os.path.join(os.path.dirname(__file__), 'results', 'normal.txt')
    with open(normal_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(str(categorized_sentences[0]))
    warning_path = os.path.join(os.path.dirname(__file__), 'results', 'warning.txt')
    with open(warning_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(str(categorized_sentences[1]))
    danger_path = os.path.join(os.path.dirname(__file__), 'results', 'danger.txt')
    with open(danger_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(str(categorized_sentences[2]))
    normal_summary_path = os.path.join(os.path.dirname(__file__), 'results', 'normal_summary.txt')
    with open(normal_summary_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(str(categorized_sentences[3]))
    warning_summary_path = os.path.join(os.path.dirname(__file__), 'results', 'warning_summary.txt')
    with open(warning_summary_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(str(categorized_sentences[4]))
    danger_summary_path = os.path.join(os.path.dirname(__file__), 'results', 'danger_summary.txt')
    with open(danger_summary_path, 'w', encoding='utf-8', errors='ignore') as f:
        f.write(str(categorized_sentences[5]))
    return categorized_sentences

if __name__ == '__main__':
    tos_path = os.path.join(os.path.dirname(__file__), 'tos.txt')
    with open(tos_path, 'r', encoding='utf-8', errors='ignore') as f:
        tos = f.read()
    app = sys.argv[1]
    analysis = analyse_tos(tos, app)