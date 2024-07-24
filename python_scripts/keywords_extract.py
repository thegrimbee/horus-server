from data import get_data
from nltk import ngrams, FreqDist

def extract_keywords(data=None, n=3):
    if data is None:
        data = get_data()
    harmful_sentences = data[data['Harm Level'] > 0]['Sentence']
    harmful_sentences = [sentence.lower() for sentence in harmful_sentences]
    combined_harmful_sentences = ' '.join(harmful_sentences).split()
    harmful_ngrams = {i: list(ngrams(combined_harmful_sentences, i)) for i in range(1, n + 1)}
    harmful_ngrams = {i: map(lambda x: ' '.join(x), harmful_ngrams[i]) for i in range(1, n + 1)}
    harmful_freq_dist = {i: FreqDist(harmful_ngrams[i]) for i in range(1, n + 1)}
    harmless_sentences = data[data['Harm Level'] == 0]['Sentence']
    harmless_sentences = [sentence.lower() for sentence in harmless_sentences]
    combined_harmless_sentences = ' '.join(harmless_sentences).split()
    harmless_ngrams = {i: list(ngrams(combined_harmless_sentences, i)) for i in range(1, n + 1)}
    harmless_ngrams = {i: map(lambda x: ' '.join(x), harmless_ngrams[i]) for i in range(1, n + 1)}
    harmless_freq_dist = {i: FreqDist(harmless_ngrams[i]) for i in range(1, n + 1)}
    normalised_harmful_freq_dist = {i: {word: freq / len(combined_harmful_sentences) 
                                    for word, freq in harmful_freq_dist[i].items()}
                                    for i in range(1, n + 1)}
    normalised_harmless_freq_dist = {i: {word: freq / len(combined_harmless_sentences)
                                    for word, freq in harmless_freq_dist[i].items()}
                                    for i in range(1, n + 1)}
    total_freq_dist = {i: harmless_freq_dist[i] + harmful_freq_dist[i] for i in range(1, n + 1)}
    ratio_list = {i: [] for i in range(1, n + 1)}
    harmful_keywords = []
    
    for i in range(1, n + 1):    
        for word in normalised_harmful_freq_dist[i].keys():
            if word in normalised_harmless_freq_dist[i]:
                value = normalised_harmful_freq_dist[i][word] / normalised_harmless_freq_dist[i][word]
                value *= len(word.split()) ** 2 if value > 1 else 1 / (len(word.split()) ** 2)
                ratio_list[i].append([word, value])
                if value > 3.5:
                    harmful_keywords.append(word)
    ratio_list = {i: sorted(ratio_list[i], key=lambda x: x[1], reverse=True) for i in range(1, n + 1)}
    ratio_dict = {}
    for j in range(1, n + 1):    
        for i in range(90-15*j):
            ratio_dict[ratio_list[j][i][0]] = ratio_list[j][i][1]
            ratio_dict[ratio_list[j][-i-1][0]] = 1 / ratio_list[j][-i-1][1]
    harmful_keywords = sorted(harmful_keywords, key=lambda x: total_freq_dist[len(x.split())][x], reverse=True) 
    return harmful_keywords, ratio_dict
