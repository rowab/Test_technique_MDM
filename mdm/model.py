import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('dataset.csv')
df.head()

df['activity']= df['activity'].map({'deco':0, 'meuble':1, 'mlp':2})
  
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
  

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
  
classifier = KNeighborsClassifier(n_neighbors= 5, metric= 'euclidean')
classifier.fit(X_train, y_train)
  
y_pred = classifier.predict(X_test)
  
score = accuracy_score(y_test, y_pred)

pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
