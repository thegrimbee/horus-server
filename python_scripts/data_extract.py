from os import path
import pandas as pd

def extract_data(appName):
    csv_path = path.join(path.dirname(__file__), '../scans.csv')
    data = pd.read_csv(csv_path)
    app_data = data[data['App'] == appName]
    categorized_sentences = app_data.iloc[0].tolist()[1:]
    extracted_data = {
        "sentence": [],
        "level": [],
    }
    for i in range(3):
        if type(categorized_sentences[i]) != str or not categorized_sentences[i]:
            continue
        for sentence in categorized_sentences[i].split('.'):
            extracted_data["sentence"].append(sentence)
            extracted_data["level"].append(i)
    output_path = path.join(path.dirname(__file__), '../data.csv')
    pd.DataFrame(extracted_data).to_csv(output_path, index=False)
    with open(path.join(path.dirname(__file__), '..', 'data.txt'), 'w', errors='ignore', encoding='utf-8') as file:
        for sentence, level in zip(extracted_data['sentence'], extracted_data['level']):
            sentence = sentence.strip('\"')
            file.write(f'{sentence},{level}\n')

extract_data('Steam')