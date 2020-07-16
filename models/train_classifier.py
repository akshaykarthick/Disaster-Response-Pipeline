import sys
import json
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import pickle
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn import preprocessing
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'wordnet'])
le = preprocessing.LabelEncoder()
import warnings
warnings.simplefilter('ignore')
import subprocess


def load_data(database_filepath):
   
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    # listing the columns
    category_names = list(np.array(y.columns))

    return X, y, category_names

def tokenize(text):
    
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())

    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:

        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
   
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'vect__min_df': [1, 5],
                'tfidf__use_idf': [True, False],
                'clf__estimator__n_estimators': [10, 25],
                'clf__estimator__min_samples_split': [5, 10]}

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, y_test, category_names):
    
    y_pred = model.predict(X_test)
    y_pred_df = pd.DataFrame(y_pred, columns = y_test.columns)
    for column in y_test.columns:
        print('------------------------------------------------------\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(y_test[column],y_pred_df[column]))


def save_model(model, model_filepath):
    
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
