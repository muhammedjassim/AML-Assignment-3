import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.tokenize import  word_tokenize
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib



# Loading data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

data = load_data('./data/emails.csv')

# Preprocessing data
def preprocess_data(data):
    data['text'] = data['text'].apply(lambda x: x.lower())  # lowercasing
    data['text'] = data['text'].apply(lambda x: x.translate(str.maketrans('', '', string.punctuation))) # punctuation removal
    data['text'] = data['text'].apply(lambda x: word_tokenize(x))   # tokenization
    stop_words = set(stopwords.words('english'))
    data['text'] = data['text'].apply(lambda tokens: [token for token in tokens if token not in stop_words])    # stop-word removal
    stemmer = PorterStemmer()
    data['text'] = data['text'].apply(lambda tokens: [stemmer.stem(token) for token in tokens]) # stemming
    data['text'] = data['text'].apply(lambda tokens: ' '.join(tokens))  # joining preprocessed tokens to form text

    return data

data = preprocess_data(data)

# Splitting data into train/validation/test
def split_data(data):
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data

train_data, test_data = split_data(data)

# Vextorization
vectorizer = TfidfVectorizer()

x_train = vectorizer.fit_transform(train_data['text'])
# x_test = vectorizer.transform(test_data['text'])

y_train = train_data['spam']

# Logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

#Saving the trained model and vectorizer
joblib.dump({'model': model, 'vectorizer': vectorizer}, 'trained_model.joblib')