from data import get_data
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import ADASYN
from textblob import TextBlob
from collections import defaultdict
# import gensim.downloader as api
import math
import xgboost as xgb
import spacy
import numpy as np
import pickle
import os
import time
nlp = spacy.load('en_core_web_sm')

def custom_loss(true, pred):
    if true == 2 and pred == 1:
        return 9
    elif true == 2 and pred == 0:
        return -15
    elif true == 1 and pred == 2:
        return 6
    elif true == 1 and pred == 0:
        return -5
    elif true == 0 and pred == 1:
        return -1.5
    elif true == 0 and pred == 2:
        return -12.5
    elif true == pred:
        return 10 # keep this constant

def grid_search_custom_loss(y_true, y_pred):
    score = 0
    for true, pred in zip(y_true, y_pred):
        score += custom_loss(true, pred)
    return score

class POSTagFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, weight=1):
        self.vectorizer = CountVectorizer()
        self.weight = weight
    
    def fit(self, X, y=None):
        pos_tags = [" ".join([token.pos_ for token in nlp(doc)]) for doc in X]
        self.vectorizer.fit(pos_tags)
        return self
    
    def transform(self, X):
        pos_tags = [" ".join([token.pos_ for token in nlp(doc)]) for doc in X]
        transformed_data = self.vectorizer.transform(pos_tags)
        if (self.weight != 1):
            transformed_data = transformed_data.multiply(self.weight)
        return transformed_data

class NERFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, weight=1):
        self.vectorizer = DictVectorizer(sparse=True)
        self.weight = weight
        # self.fitted = False
    
    def fit(self, X, y=None):
        label_dicts = []
        for doc in X:
            labels = [ent.label_ for ent in nlp(doc).ents]
            label_freq = {label: labels.count(label) for label in set(labels)}
            label_dicts.append(label_freq)
        self.vectorizer.fit(label_dicts)
        return self
    
    def transform(self, X):
        label_dicts = []
        for doc in X:
            labels = [ent.label_ for ent in nlp(doc).ents]
            label_freq = {label: labels.count(label) for label in set(labels)}
            label_dicts.append(label_freq)
        transformed_data = self.vectorizer.transform(label_dicts)
        if (self.weight != 1):
            transformed_data = transformed_data.multiply(self.weight)
        return transformed_data


class DependencyFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, weight=1):
        self.vectorizer = DictVectorizer(sparse=True)
        self.weight = weight
        # self.fitted = False
    
    def fit(self, X, y=None):
        rel_dicts = []
        for doc in X:
            dep_rels = ["{}_{}".format(token.dep_, token.head.text) for token in nlp(doc)]
            rel_freq = {rel: dep_rels.count(rel) for rel in set(dep_rels)}
            rel_dicts.append(rel_freq)
        self.rel_dicts = rel_dicts
        self.vectorizer.fit(rel_dicts)
        return self
    
    def transform(self, X):
        rel_dicts = []
        for doc in X:
            dep_rels = ["{}_{}".format(token.dep_, token.head.text) for token in nlp(doc)]
            rel_freq = {rel: dep_rels.count(rel) for rel in set(dep_rels)}
            rel_dicts.append(rel_freq)
        self.rel_dicts = rel_dicts
        transformed_data = self.vectorizer.transform(rel_dicts)
        if (self.weight != 1):
            transformed_data = transformed_data.multiply(self.weight)
        return transformed_data
    
class SentimentFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, weight=1):
        self.weight = weight
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        polarities = []
        subjectivities = []
        for doc in X:
            blob = TextBlob(doc)
            polarities.append(blob.sentiment.polarity)
            subjectivities.append(blob.sentiment.subjectivity)
        transformed_data = np.array([polarities, subjectivities]).T
        if (self.weight != 1):
            transformed_data *= self.weight
        return transformed_data

class KeywordFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, keywords, weight=1, keyword_weights={}):
        self.keywords = keywords
        self.keyword_weights = keyword_weights
        self.weight = weight
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        transformed_data = [[doc.count(keyword) * self.keyword_weights.get(keyword, 1) for keyword in self.keywords] for doc in X]
        if (self.weight != 1):
            transformed_data = [[count * self.weight for count in doc] for doc in transformed_data]
        return transformed_data

class ClauseContextFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, keywords, weight=1, keyword_weights={}):
        self.keywords = keywords
        self.keyword_weights = keyword_weights
        self.vectorizer = DictVectorizer(sparse=True)
        self.weight = weight

    def fit(self, X, y=None):
        context_dicts = self._generate_context_dicts(X)
        self.vectorizer.fit(context_dicts)
        return self

    def transform(self, X):
        context_dicts = self._generate_context_dicts(X)
        transformed_data = self.vectorizer.transform(context_dicts)
        if (self.weight != 1):
            transformed_data = transformed_data.multiply(self.weight)
        return transformed_data

    def _generate_context_dicts(self, X):
        context_dicts = []
        for doc in X:
            doc_nlp = nlp(doc)
            clauses = [sent for sent in doc_nlp.sents]
            context_dict = defaultdict(int)
            for clause in clauses:
                clause_text = clause.text
                for keyword in self.keywords:
                    if keyword in clause_text:
                        weight = self.keyword_weights.get(keyword, 1)
                        context_dict[keyword] += weight
                        # You can also add more contextual information here
                        context_dict[f'{keyword}_context'] = clause_text
            context_dicts.append(context_dict)
        return context_dicts
    
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
    
minilm_path = os.path.join(os.path.dirname(__file__), '../ai_models/minilm')
# Custom transformer for Sentence Transformers
class SentenceTransformerFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', weight=1):
        self.model = SentenceTransformer(minilm_path)
        self.model_name = model_name
        self.weight = weight
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        embeddings = self.model.encode(X if type(X) == list else X.tolist(), convert_to_tensor=False)
        transformed_data = np.array(embeddings)
        if (self.weight != 1):
            transformed_data = transformed_data * self.weight
        return transformed_data

class CustomXGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, params=None, num_boost_round=200, learning_rate=0.05, max_depth=5, 
                 hessian_penalty=1, sqrt=False, penalty='factor'):
        self.learning_rate = learning_rate
        self.hessian_penalty = hessian_penalty
        self.max_depth = max_depth
        self.penalty = penalty
        params['learning_rate'] = self.learning_rate
        params['max_depth'] = self.max_depth
        self.params = params if params is not None else {}
        self.num_boost_round = num_boost_round
        self.model = None
        self.sqrt = sqrt
        self.classes_ = np.array([0, 1, 2])

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def xgb_custom_loss_calculate(self, y_true, y_pred, hessian_penalty=1, sqrt=False, penalty='factor'):
        num_classes = 3
        num_samples = y_true.shape[0]
        
        # Reshape y_pred to [num_samples, num_classes]
        y_pred = y_pred.reshape(num_samples, num_classes)
        
        # Apply softmax to get predicted probabilities
        y_pred_prob = self.softmax(y_pred)
        
        # Initialize gradient and hessian
        gradient = np.zeros_like(y_pred)
        hessian = np.zeros_like(y_pred)
        # Compute gradient and hessian for each class
        for i in range(num_classes):
            y_true_i = (y_true == i).astype(float)
            gradient[:, i] = y_pred_prob[:, i] - y_true_i
            hessian[:, i] = y_pred_prob[:, i] * (1.0 - y_pred_prob[:, i])
        penalty_conditions = [
            ((y_true == 2) & (np.argmax(y_pred_prob, axis=1) == 0), 0, 20),
            ((y_true == 1) & (np.argmax(y_pred_prob, axis=1) == 0), 0, 6.5),
            ((y_true == 2) & (np.argmax(y_pred_prob, axis=1) == 1), 1, 0.15),
            ((y_true == 0) & (np.argmax(y_pred_prob, axis=1) == 1), 1, 1),
            ((y_true == 1) & (np.argmax(y_pred_prob, axis=1) == 2), 2, 1.5),
            ((y_true == 0) & (np.argmax(y_pred_prob, axis=1) == 2), 2, 8),
            ((y_true == 2) & (np.argmax(y_pred_prob, axis=1) == 2), 2, 0.25),
        ]
        
        if penalty == 'factor':
            
            # Original penalty factors for the gradient
            penalty_factors_gradient = np.ones_like(gradient)

            # New penalty factors for the Hessian, initially the same as for the gradient
            penalty_factors_hessian = np.ones_like(hessian)

            # Define a reduction factor for the Hessian penalties
            hessian_penalty_reduction_factor = hessian_penalty  # Example: Reduce Hessian penalties by half

            for condition, pred_class, factor in penalty_conditions:
                penalty_factors_gradient[condition, pred_class] *= factor
                # Apply reduced penalty to Hessian
                if sqrt:
                    adjusted_factor = np.sqrt(factor) * factor  # Example of a non-linear adjustment
                    penalty_factors_hessian[condition, pred_class] *= adjusted_factor
                else:
                    penalty_factors_hessian[condition, pred_class] *= factor * hessian_penalty_reduction_factor
            
            # Apply penalties to gradients and hessians separately
            gradient *= penalty_factors_gradient
            hessian *= penalty_factors_hessian
        elif penalty == 'exp':
            for condition, class_index, target in penalty_conditions:
                # Calculate the distance from the target probability
                distance = np.abs(y_pred_prob[:, class_index] * target)
                
                # Apply exponential decay based on the distance
                penalty = np.exp(-1 / distance)
                
                # Find indices where the condition is True
                condition_indices = np.where(condition)
                
                # Adjust the gradient and Hessian for these indices
                gradient[condition_indices, class_index] += penalty[condition_indices]
                hessian[condition_indices, class_index] += penalty[condition_indices] * hessian_penalty
        return gradient.flatten(), hessian.flatten()

    def xgb_custom_loss(self, y_pred, dtrain, hessian_penalty=1, sqrt=False, penalty='factor'):
        y_true = dtrain.get_label()
        gradient, hessian = self.xgb_custom_loss_calculate(y_true, y_pred, hessian_penalty, sqrt, penalty)
        return gradient, hessian
    
    def fit(self, X, y):
        dtrain = xgb.DMatrix(data=X, label=y)
        customised_hessian_loss_func = lambda y_pred, dtrain: self.xgb_custom_loss(y_pred, dtrain, 
                                                                              self.hessian_penalty, self.sqrt,
                                                                              self.penalty)
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            obj=customised_hessian_loss_func
        )
        return self

    def predict(self, X):
        dtest = xgb.DMatrix(data=X)
        y_pred_prob = self.model.predict(dtest)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_pred = self.classes_[y_pred]
        return y_pred

    def predict_proba(self, X):
        dtest = xgb.DMatrix(data=X)
        y_pred_prob = self.model.predict(dtest)
        return y_pred_prob

class CustomModel:
    def __init__(self, model, keywords):
        self.model = model

    def predict(self, X):
        passed = False
        for word in keywords:
            if word in X:
                passed = True
                break
        if passed:
            return self.model.predict(X)
        
# Define your XGBoost parameters
xgb_params = {
    'objective': 'multi:softprob',
    'num_class': 3,
    'eval_metric': 'mlogloss',
    'learning_rate': 0.05,
    'max_depth': 5
}

if __name__ == '__main__':
    custom_scorer = make_scorer(grid_search_custom_loss, greater_is_better=True)
    data = get_data()
    train_data, val_data = train_test_split(data, test_size=0.2, stratify=data['Harm Level'])

    # Extract features and labels for training data
    X_train = train_data['Sentence']
    Y_train = train_data['Harm Level']

    # Extract features and labels for validation data
    X_test = val_data['Sentence']
    Y_test = val_data['Harm Level']

    length = len(X_train)
    keywords = ["privacy", "data", "rights", "terminate", "warranty", "loss", "age", "personal", 
                "information", "location", "termination", "cookies", "security", "third-party",
                "personal data", "personal information", "personal details", "collect information",
                "without notice", "collect"]
    keyword_weights = {
        "personal data": 5,
        "personal information": 5,
        "personal details": 5,
        "collect information": 5,
        "without notice": 5,
        "loss": 3,
        "collect": 2,
        "location": 2,
        "personal": 2,
        "data": 1.5,
        "rights": 1.5,
    }
    pipeline = ImbPipeline([
        # ('smote', SMOTE(sampling_strategy='auto')),
        ('features', FeatureUnion([
            ('keywords', KeywordFeatures(keywords=keywords, weight=0.25, keyword_weights=keyword_weights)),
            ('pos_tags', POSTagFeatures(weight=0.7)),
            ('ner', NERFeatures(weight=0.95)),
            ('dependency', DependencyFeatures(weight=0.7)),
            ('clause_context', ClauseContextFeatures(keywords=keywords, weight=0.6, keyword_weights={})),
            ('sentiment', SentimentFeatures(weight=0.3)),
            ('sentence_transformer', SentenceTransformerFeatures(weight=0.8)),
            # ('glove', GloVeEmbeddings(model_name='glove-wiki-gigaword-100'))
        ])),
        ('adasyn', ADASYN(sampling_strategy='auto', n_neighbors=3)),
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', CustomXGBClassifier(params=xgb_params, num_boost_round=250, learning_rate=0.075, max_depth=8,
                                    hessian_penalty=0.2, sqrt=False, penalty='factor')),
        # ('clf', GradientBoostingClassifier(
        #     n_estimators=200, 
        #     max_depth=5, 
        #     learning_rate=0.05, 
        #     min_samples_split=6, 
        #     min_samples_leaf=2, 
        #     subsample=0.9, 
        #     max_features=None, 
        #     warm_start=True
        # )), 
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
    # param_grid = {
    #     'features__keywords__weight': [0.2, 0.4], # low
    #     # 'features__keywords__keyword_weights': [keyword_weights],
    #     # 'features__pos_tags__weight': [0.7], # low
    #     # 'features__ner__weight': [0.95],
    #     # 'features__dependency__weight': [0.7], # low
    #     # 'features__clause_context__weight': [0.6], # low
    #     # 'features__clause_context__keyword_weights': [{}],
    #     # 'features__sentiment__weight': [0.3], # low
    #     # 'features__sentence_transformer__weight': [0.8], # high
    #     # 'clf__num_boost_round': [250],
    #     # 'clf__learning_rate': [0.075],
    #     # 'clf__max_depth': [7],
    #     # 'clf__hessian_penalty': [1],
    #     # 'clf__sqrt': [True],
    # }


    # stratified_kfold = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    # grid_search = GridSearchCV(pipeline, param_grid, cv=stratified_kfold, n_jobs=1, verbose=2, scoring=custom_scorer)
    # grid_search.fit(X_train, Y_train)
    # model = grid_search.best_estimator_
    # print(grid_search.best_params_)
    # print(grid_search.best_score_ * 2 / length)
    print('testing1')
    start = time.time()
    pipeline.fit(X_train, Y_train)
    model = pipeline

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
    end = time.time()
    print(end - start)