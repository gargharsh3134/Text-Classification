import pandas as pd
import numpy as np
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from imblearn.over_sampling import SMOTE
from sklearn import svm


dataset = pd.read_csv('combined3.csv', encoding = 'ISO-8859-1')
y = dataset.iloc[:,1]
X= dataset.iloc[:,0]

corpus = []
for i in range(0, len(X)):
    review = re.sub(r'\W', ' ', str(X[i]))
    review = review.lower()
    review = re.sub(r'\s+[a-z]\s+',' ', review)
    review = re.sub(r'^[a-z]\s', ' ', review)
    review = re.sub(r'\s+', ' ', review)
    review = re.sub(r'â' , ' ' , review)
    corpus.append(review)
    


from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=2000, min_df = 3, max_df=0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()


from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X = transformer.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0 )

#oversampler = SMOTE(random_state = 0)
#X_train, y_train = oversampler.fit_sample(X1_train,y1_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators= 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=2000, min_df = 3, max_df=0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()

with open('classifier.pickle', 'wb') as f:
    pickle.dump(classifier, f)


with open('Tfidfmodel.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)



with open('classifier.pickle', 'rb') as f:
    clf = pickle.load(f)



with open('Tfidfmodel.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

    
    
    
from flask import Flask, render_template, request
with open('classifier.pickle', 'rb') as f:
    clf = pickle.load(f)



with open('Tfidfmodel.pickle', 'rb') as f:
    Tfidf = pickle.load(f)

app = Flask(__name__)
 
@app.route('/')
def sentence():
   return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
        result = str(request.form['Name'])
        #result = str(request.form)  
        fw = open('sample.txt', 'w')
        fw.write(result)
        fw.close()
        result = Tfidf.transform(open('sample.txt','r')).toarray()
        result = int(clf.predict(result))
        if (result is 0):
            result = 'negative'
        elif (result is 1):
            result = 'positive'
        elif (result is 2):
            result = 'neutral'
        #elif (result is 3):
        #   result = 'suggestion'
        #elif (result is 4) :
         #   result = 'neutral'
        return render_template("result.html",result = result)

if __name__ == '__main__':
    app.run()




