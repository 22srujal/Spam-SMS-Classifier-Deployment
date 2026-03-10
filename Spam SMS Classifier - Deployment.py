from __future__ import annotations

import pickle

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

DATASET_PATH = 'Spam SMS Collection'
VECTORIZER_PKL = 'cv-transform.pkl'
MODEL_PKL = 'spam-sms-mnb-model.pkl'


def main() -> None:
  df = pd.read_csv(
    DATASET_PATH,
    sep='\t',
    names=['label', 'message'],
    encoding='latin-1',
  )

  x_text = df['message'].astype(str)
  y = df['label'].map({'ham': 0, 'spam': 1}).astype(int)

  x_train_text, x_test_text, y_train, y_test = train_test_split(
    x_text,
    y,
    test_size=0.20,
    random_state=0,
    stratify=y,
  )

  cv = CountVectorizer(
    max_features=2500,
    stop_words='english',
  )
  x_train = cv.fit_transform(x_train_text)
  x_test = cv.transform(x_test_text)

  classifier = MultinomialNB(alpha=0.3)
  classifier.fit(x_train, y_train)

  y_pred = classifier.predict(x_test)
  print('Accuracy:', accuracy_score(y_test, y_pred))

  with open(VECTORIZER_PKL, 'wb') as vectorizer_file:
    pickle.dump(cv, vectorizer_file)

  with open(MODEL_PKL, 'wb') as model_file:
    pickle.dump(classifier, model_file)


if __name__ == '__main__':
  main()