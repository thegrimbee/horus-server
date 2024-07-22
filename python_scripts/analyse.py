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
from ai import SentenceTransformerFeatures, POSTagFeatures, NERFeatures, \
KeywordFeatures, DependencyFeatures, SentimentFeatures, CustomXGBClassifier, \
ClauseContextFeatures
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
        elif name == 'ClauseContextFeatures':
            from ai import ClauseContextFeatures
            return ClauseContextFeatures
        elif name == 'CustomXGBClassifier':
            from ai import CustomXGBClassifier
            return CustomXGBClassifier

        return super().find_class(module, name)

def summarize(text):
    if text == '':
        return text
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    ouputs = model.generate(inputs, max_length=150, min_length=25, length_penalty=5.0, num_beams=2, early_stopping=True)
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
    scanned_apps = list(scans['App'].values)
    scanned_apps = [app.lower() for app in scanned_apps]
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
        driver.implicitly_wait(10)
        p_elements = driver.find_elements(By.TAG_NAME, 'p')
        LENGTH_CRITERIA = 30
        WORD_CRITERIA = 5
        SENTENCES_CRITERIA = 5
        print("Received elements")
        sentences = []
        for element in p_elements:
            if len(sentences) >= 350:
                break
            try:
                if len(element.text) > LENGTH_CRITERIA: # and len(element.text.split()) > WORD_CRITERIA:
                    sentences.extend(element.text.split('.'))
            except:
                continue
        if len(sentences) < SENTENCES_CRITERIA:
            div_elements = driver.find_elements(By.TAG_NAME, 'div')
            for element in div_elements:
                if len(sentences) >= 350:
                    break
                try:
                    if len(element.text) > LENGTH_CRITERIA: # and len(element.text.split()) > WORD_CRITERIA:
                        sentences.extend(element.text.split('.'))

                except:
                    continue
        print("tos:", sentences[:5])
        driver.quit()
    else:
        sentences = tos.split('.')
    print("Finished online processing")
    memory_use = current_process.memory_info().rss
    print(f"Current memory usage: {memory_use / 1024**2:.2f} MB")
    categorized_sentences = ["","",""]
    if is_scanned:
        print('App found in scans.csv')
        categorized_sentences = scans[scans['App'].str.lower() == app.lower()].iloc[0].tolist()[1:]
    if not check_valid(categorized_sentences):
        model_path = os.path.join(os.path.dirname(__file__), '../ai_models/model3.pkl')
        with open(os.path.join(model_path), 'rb') as file:
            model = CustomUnpickler(file).load()
        print("Loaded model")
        memory_use = current_process.memory_info().rss
        print(f"Current memory usage: {memory_use / 1024**2:.2f} MB")
        categorized_sentences = ["","",""]
        for sentence in sentences[:350]:
            categorized_sentences[predict(sentence, model)] += "\n" + sentence
        memory_use = current_process.memory_info().rss
        print("Finished analysing")
        print(f"Current memory usage: {memory_use / 1024**2:.2f} MB")
        for i in range(3):
            categorized_sentences.append(summarize(categorized_sentences[i]))
        print("Finished summarizing")
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