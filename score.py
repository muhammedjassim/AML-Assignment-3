from typing import Tuple

def score(text: str, model_data, threshold: float) -> Tuple[bool, float]:

    model = model_data['model']
    vectorizer = model_data['vectorizer']
    
    text_vector = vectorizer.transform([text])
    
    propensity = model.predict_proba(text_vector)[0][1]
    prediction = 1 if propensity >= threshold else 0
    
    return prediction, propensity