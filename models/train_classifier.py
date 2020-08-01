"""
Training Classifier

"""
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
import pickle
nltk.download('stopwords')


def load_data(database_filepath):
   """
    Function to load Data from the DB
    
    Inputs:
       Path to SQLite destination database  -> database_filepath  

    Output:
        dataframe of features = X
        dataframe of labels = Y
        Categories name = category_names
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('df', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])

    return X, Y, category_names



def tokenize(text):
     """
    Tokenization function to process the text data

    """
    # Normalize the text
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text.lower())
    
    # Extracting the word tokens from the given text
    words = word_tokenize(text)
    
    # Remove Stopwords
    words = [w for w in words if w not in stopwords.words('english')]
    
    #Lemmanitizer to remove derivational forms of a word
    lemmatizer = WordNetLemmatizer()
    clean = [lemmatizer.lemmatize(w, pos='n').strip() for w in words]
    clean = [lemmatizer.lemmatize(w, pos='v').strip() for w in clean]
    
     # Return the list of clean tokens
    return clean


def build_model():
    '''
    Building a Machine Learning pipeline with RandomForest classifier GridSearch.
    
    '''
    pipeline = Pipeline([
                        ('vect', CountVectorizer(tokenizer=tokenize)),
                        ('tfidf', TfidfTransformer()),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])

    parameters = {'clf__estimator__n_estimators': [50, 60],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    # Returns grid search output
    return cv

#Evaluate the model performance
def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model performance.
    
    Input:
        model: Evaluation model
        X_test: Test data containing features
        Y_test: True lables for Test data
        category_names: Labels for 36 categories
    Output:
        Each catagories accuracy and classification report
    '''
    #predict using the test data
    Y_pred = model.predict(X_test)

    # Calculate the classification and accuracy for each of them.
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))

#Dump the pickle file
def save_model(model, model_filepath):
    '''
    Save model as a pickle file
    
    Input:
        model: Model to be saved
        model_filepath: output pick file path
        
    Output:
        Dumped pickle file
    '''
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
