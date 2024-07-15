import sys
import pickle
import os
import pandas as pd
from googlesearch import search
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
import warnings
import psutil

webdriver_options = Options()
webdriver_options.add_argument("--headless")

# Get current process ID
pid = os.getpid()
# Get the process object using PID
current_process = psutil.Process(pid)

# Ignore FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

from transformers import AutoTokenizer, T5ForConditionalGeneration
from ai import SentenceTransformerFeatures, POSTagFeatures, NERFeatures, KeywordFeatures, DependencyFeatures, SentimentFeatures
# Custom unpickler example

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'SentenceTransformerFeatures':
            from ai import SentenceTransformerFeatures
            return SentenceTransformerFeatures
        elif name == 'POSTagFeatures':
            from ai import POSTagFeatures
            return POSTagFeatures
        elif name == 'NERFeatures':
            from ai import NERFeatures
            return NERFeatures
        elif name == 'KeywordFeatures':
            from ai import KeywordFeatures
            return KeywordFeatures
        elif name == 'DependencyFeatures':
            from ai import DependencyFeatures
            return DependencyFeatures
        elif name == 'SentimentFeatures':
            from ai import SentimentFeatures
            return SentimentFeatures

        return super().find_class(module, name)

def summarize(text):
    if text == '':
        return text
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    ouputs = model.generate(inputs, max_length=150, min_length=75, length_penalty=5.0, num_beams=2, early_stopping=True)
    return tokenizer.decode(ouputs[0], skip_special_tokens=True)

def predict(sentence, model):
    return model.predict([sentence])[0]

def check_valid(c_s):
    total = 0
    for i in c_s[:3]:
        if type(i) == str:
            total += len(i)
    return total > 10

def analyse_tos(tos, app="", url=""):
    print(f'Analysing {app}')
    scans_path = os.path.join(os.path.dirname(__file__), '../scans.csv')
    scans = pd.read_csv(scans_path)
    scanned_apps = map(scans['App'].values, lambda x: x.lower())
    is_scanned = app.lower() in scanned_apps
    if not is_scanned and tos.strip()== '':
        print("No terms of service found for " + app + ". Searching the web...")
        if url == '':
            tos_urls = search(app + " terms of service", num=1, stop=1)
            for i in tos_urls:
                url = i
        print("url:", url)
        memory_use = current_process.memory_info().rss
        print(f"Current memory usage: {memory_use / 1024**2:.2f} MB")
        driver = webdriver.Firefox(options=webdriver_options)
        driver.get(url)
        p_elements = driver.find_elements(By.TAG_NAME, 'p')
        for i in p_elements:
            tos += i.text
        print("tos:", tos[:50])
        driver.quit()

    memory_use = current_process.memory_info().rss
    print(f"Current memory usage: {memory_use / 1024**2:.2f} MB")
    categorized_sentences = ["","",""]
    if is_scanned:
        print('App found in scans.csv')
        categorized_sentences = scans[scans['App'] == app].iloc[0].tolist()[1:]
    if not check_valid(categorized_sentences):
        sentences = tos.split('.')
        model_path = os.path.join(os.path.dirname(__file__), '../ai_models/model2.pkl')
        with open(os.path.join(model_path), 'rb') as file:
            model = CustomUnpickler(file).load()
        memory_use = current_process.memory_info().rss
        print(f"Current memory usage: {memory_use / 1024**2:.2f} MB")
        categorized_sentences = ["","",""]
        for sentence in sentences:
            categorized_sentences[predict(sentence, model)] += "\n" + sentence
        memory_use = current_process.memory_info().rss
        print(f"Current memory usage: {memory_use / 1024**2:.2f} MB")
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
        if check_valid(categorized_sentences):
            scans.to_csv(scans_path, index=False)
    counter = 0
    for sentence in categorized_sentences:
        print('Sentence' + str(counter))
        counter += 1
        if (type(sentence) == str):
            print(sentence[:50])
    memory_use = current_process.memory_info().rss
    print(f"Current memory usage: {memory_use / 1024**2:.2f} MB")
    return categorized_sentences

if __name__ == '__main__':
    app = sys.argv[1]
    url = sys.argv[2]
    print(analyse_tos("", app, url))