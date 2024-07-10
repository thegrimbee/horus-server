from data import get_data
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.feature_extraction import DictVectorizer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import ADASYN
from textblob import TextBlob
# import gensim.downloader as api
import spacy
import numpy as np
import pickle
import os

# Custom transformer to extract TextBlob features
# class TextBlobFeatures(BaseEstimator, TransformerMixin):
#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         polarities = []
#         subjectivities = []
#         for text in X:
#             blob = TextBlob(text)
#             polarities.append(blob.sentiment.polarity)
#             subjectivities.append(blob.sentiment.subjectivity)
#         return np.array([polarities, subjectivities]).T
nlp = spacy.load('en_core_web_sm')

class POSTagFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = CountVectorizer()

    def fit(self, X, y=None):
        pos_tags = [" ".join([token.pos_ for token in nlp(doc)]) for doc in X]
        # Fit the vectorizer with the extracted POS tags
        self.vectorizer.fit(pos_tags)
        return self
    
    def transform(self, X):
        pos_tags = [" ".join([token.pos_ for token in nlp(doc)]) for doc in X]
        return self.vectorizer.transform(pos_tags)

class NERFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)
        self.fitted = False

    def fit(self, X, y=None):
        label_dicts = []
        for doc in X:
            labels = [ent.label_ for ent in nlp(doc).ents]
            label_freq = {label: labels.count(label) for label in set(labels)}
            label_dicts.append(label_freq)
        
        # Fit the DictVectorizer to the NER label frequencies
        self.vectorizer.fit(label_dicts)
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("The transformer has not been fitted yet.")
        
        # Convert NER labels to a list of dictionaries with label frequencies
        label_dicts = []
        for doc in X:
            labels = [ent.label_ for ent in nlp(doc).ents]
            label_freq = {label: labels.count(label) for label in set(labels)}
            label_dicts.append(label_freq)
        
        # Transform the list of label frequency dictionaries to a numeric matrix
        return self.vectorizer.transform(label_dicts)

class DependencyFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.vectorizer = DictVectorizer(sparse=True)
        self.fitted = False

    def fit(self, X, y=None):
        # Convert dependency tags to a list of dictionaries with tag frequencies
        tag_dicts = []
        for doc in X:
            tags = [token.dep_ for token in nlp(doc)]
            tag_freq = {tag: tags.count(tag) for tag in set(tags)}
            tag_dicts.append(tag_freq)
        
        # Fit the DictVectorizer to the dependency tag frequencies
        self.vectorizer.fit(tag_dicts)
        self.fitted = True
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("The transformer has not been fitted yet.")
        
        # Convert dependency tags to a list of dictionaries with tag frequencies
        tag_dicts = []
        for doc in X:
            tags = [token.dep_ for token in nlp(doc)]
            tag_freq = {tag: tags.count(tag) for tag in set(tags)}
            tag_dicts.append(tag_freq)
        
        # Transform the list of tag frequency dictionaries to a numeric matrix
        return self.vectorizer.transform(tag_dicts)
    
class SentimentFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [[TextBlob(doc).sentiment.polarity] for doc in X]

class KeywordFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, keywords):
        self.keywords = keywords
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return [[doc.count(keyword) for keyword in self.keywords] for doc in X]

# class GloVeEmbeddings(BaseEstimator, TransformerMixin):
#     def __init__(self, model_name='glove-wiki-gigaword-100'):
#         self.model_name = model_name
    
#     def fit(self, X, y=None):
#         self.model = api.load(self.model_name)
#         return self
    
#     def transform(self, X):
#         def get_average_embedding(doc):
#             embeddings = [self.model[word] for word in doc.split() if word in self.model]
#             if embeddings:
#                 return np.mean(embeddings, axis=0)
#             else:
#                 return np.zeros(self.model.vector_size)
        
#         return np.array([get_average_embedding(doc) for doc in X])
    
# minilm_path = os.path.join(os.path.dirname(__file__), '../ai_models/minilm')
# Custom transformer for Sentence Transformers
class SentenceTransformerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.model_name = model_name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        embeddings = self.model.encode(X if type(X) == list else X.tolist(), convert_to_tensor=False)
        return np.array(embeddings)
    
def custom_loss(y_true, y_pred):
    score = 0
    for true, pred in zip(y_true, y_pred):
        if true == 2 and pred == 1:
            score += 7
        elif true == 2 and pred == 0:
            score -= 15
        elif true == 1 and pred == 2:
            score += 3
        elif true == 1 and pred == 0:
            score -= 3
        elif true == 0 and pred == 1:
            score -= 3
        elif true == 0 and pred == 2:
            score -= 11
        elif true == pred:
            score += 10 # keep this constant
    return score

if __name__ == '__main__':
    custom_scorer = make_scorer(custom_loss, greater_is_better=True)
    data = get_data()
    train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['Harm Level'])

    # Extract features and labels for training data
    X_train = train_data['Sentence']
    Y_train = train_data['Harm Level']

    # Extract features and labels for validation data
    X_test = val_data['Sentence']
    Y_test = val_data['Harm Level']

    length = len(X_train)

    pipeline = ImbPipeline([
        ('features', FeatureUnion([
            # ('tfidf', TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_df=0.5, min_df=5)),
            ('sentence_transformer', SentenceTransformerFeatures()),
            ('pos_tags', POSTagFeatures()),
            ('ner', NERFeatures()),
            ('dependency', DependencyFeatures()),
            ('sentiment', SentimentFeatures()),
            ('keywords', KeywordFeatures(keywords=["privacy", "data", "rights", "terminate", "warranty", "loss"])),
            # ('glove', GloVeEmbeddings(model_name='glove-wiki-gigaword-100'))
            #('textblob', TextBlobFeatures()),
        ])),
        # ('smote', SMOTE(sampling_strategy='auto')),
        ('adasyn', ADASYN(sampling_strategy='auto', n_neighbors=3)),
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', GradientBoostingClassifier(
            n_estimators=200, 
            max_depth=5, 
            learning_rate=0.05, 
            min_samples_split=6, 
            min_samples_leaf=2, 
            subsample=0.9, 
            max_features=None, 
            warm_start=True
        )), 
        # ('clf', RandomForestClassifier(
        #         n_estimators=200, 
        #         max_depth = None,
        #         min_samples_split = 5,
        #         max_features = 'sqrt',
        #         criterion = 'gini',
        #         max_leaf_nodes = 30,
        #         min_impurity_decrease=0.0,
        #         class_weight='balanced_subsample',
        #         bootstrap=True,
        #         min_samples_leaf=4,
        #     )),
        # ('clf', XGBClassifier(
        #         n_estimators=200, 
        #         max_depth=None,
        #         learning_rate=0.1,
        #         colsample_bytree=1.0,
        #         subsample=0.8,
        #     )),
        # ('clf', LGBMClassifier(
        #         n_estimators=200, 
        #         max_depth=None,
        #         learning_rate=0.05,
        #         colsample_bytree=1.0,
        #         subsample=0.8,
        #         num_leaves=24,
        #     )),
    ])

    # Define parameter grid for GridSearchCV
    param_grid = {
    }


    stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kfold, n_jobs=1, verbose=0, scoring=custom_scorer)
    grid_search.fit(X_train, Y_train)
    model = grid_search.best_estimator_
    print(grid_search.best_params_)
    print(grid_search.best_score_ / length)
    # print('testing1')
    # pipeline.fit(X_train, Y_train)
    # model = pipeline

    model_path = os.path.join(os.path.dirname(__file__), '../ai_models/model2.pkl')
    # Save the model to a file
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    # Predict on the test set
    Y_pred = model.predict(X_test) 
    print('testing2')
    # Evaluate the model

    accuracy = accuracy_score(Y_test, Y_pred)
    print(f'Accuracy: {accuracy}')
    print('Classification Report:')
    print(classification_report(Y_test, Y_pred))
    print(Y_pred.tolist())
    print(Y_test.tolist())
