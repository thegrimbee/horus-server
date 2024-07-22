from data import get_data
from nltk import FreqDist

def extract_keywords():
    data = get_data()
    harmful_sentences = data[data['Harm Level'] > 0]['Sentence']
    harmful_sentences = [sentence.lower() for sentence in harmful_sentences]
    combined_harmful_sentences = ' '.join(harmful_sentences).split()
    harmless_sentences = data[data['Harm Level'] == 0]['Sentence']
    harmless_sentences = [sentence.lower() for sentence in harmless_sentences]
    combined_harmless_sentences = ' '.join(harmless_sentences).split()
    harmful_freq_dist = FreqDist(combined_harmful_sentences)
    normalised_harmful_freq_dist = {word: freq / len(combined_harmful_sentences) 
                                    for word, freq in harmful_freq_dist.items()}
    normalised_harmful_freq_dist_list = sorted(normalised_harmful_freq_dist.items(), key=lambda x: x[1], reverse=True)
    harmless_freq_dist = FreqDist(combined_harmless_sentences)
    normalised_harmless_freq_dist = {word: freq / len(combined_harmless_sentences) 
                                    for word, freq in harmless_freq_dist.items()}
    normalised_harmless_freq_dist_list = sorted(normalised_harmless_freq_dist.items(), key=lambda x: x[1], reverse=True)
    ratio_list = []
    harmful_keywords = []
    for word in normalised_harmful_freq_dist.keys():
        if word in normalised_harmless_freq_dist:
            value = normalised_harmful_freq_dist[word] / normalised_harmless_freq_dist[word]
            ratio_list.append([word, value])
            if value > 1.5:
                harmful_keywords.append(word)
    ratio_list = sorted(ratio_list, key=lambda x: x[1], reverse=True)
    ratio_dict = {}
    for i in range(100):
        ratio_dict[ratio_list[i][0]] = ratio_list[i][1]
        ratio_dict[ratio_list[-i-1][0]] = ratio_list[-i-1][1]
    
    return harmful_keywords, ratio_dict