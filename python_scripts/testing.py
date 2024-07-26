from analyse import analyse_tos, check_valid, summarize
import pickle
import os

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
    
def test_analyse():
    # Limited testing is possible due to the complexity of the function
    assert analyse_tos("", "Test", "") == ["Testing00","Testing01","Testing02","TestSum0","TestSum1","TestSum2"] 
    assert check_valid(["This is just a test for the validity of the sentence",
                        "Any long enough entry should work",
                        "As long as it passes a certain length criteria",
                        "The next few columns are just summaries",
                        "They are irrelevant to check valid",
                        "",]) == True
    assert check_valid(["Short", "Sentences", "Fail", "", "", ""]) == False
    assert check_valid(["Purpose of the function", "Is for failsafe in case online scanning", "Gets empty websites",
                        "", "", ""]) == True
    model_path = os.path.join(os.path.dirname(__file__), '../ai_models/model4.pkl')
    with open(os.path.join(model_path), 'rb') as file:
        model = CustomUnpickler(file).load()
    assert model.predict(["We collect all your personal data and share them with third parties",])[0]> 0
    assert model.predict(["You have the right to terminate your account any time"])[0] == 0
    result = model.predict(["We respect your privacy"] * 100)
    assert all(0 <= item <= 2 for item in result)
    assert all(item == 0 for item in result)
    assert summarize("This is a test") == "This is a test"
    assert summarize("Short sentences do not need to get summarised") == "Short sentences do not need to get summarised"
    assert len(summarize("This is a test" * 200, True).split()) < 200
    assert len(summarize("This is a test" * 100, False).split()) < 100 

