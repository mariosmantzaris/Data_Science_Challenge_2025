import csv
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Read descriptions of products
descriptions = dict()
with open("description.txt", "r") as f:
    for line in f:
        t = line.split('|=|')
        descriptions[int(t[0])] = t[1][:-1]

# Read training data
train_data = list()
y_train = list()
with open("y_train.txt", "r") as f:
    for i,line in enumerate(f):
        t = line.split(',')
        train_data.append(int(t[0]))
        y_train.append(int(t[1][:-1]))

# Read test data
test_data = list()
with open("test.txt", "r") as f:
    for i,line in enumerate(f):
        t = line.split(',')
        test_data.append(int(t[0]))

# Retrieve descriptions of products in the training set
train_text = list()
for i in train_data:
    train_text.append(descriptions[i])

# Retrieve descriptions of products in the test set
test_text = list()
for i in test_data:
    test_text.append(descriptions[i])

# Create the training matrix. Each row corresponds to a product and each column to a word present in at least 5 descriptions
# The value of each entry in a row is equal to the tf-idf weight of that word in the corresponding domain       
vec = TfidfVectorizer(decode_error='ignore', strip_accents='unicode', min_df=5)
X_train = vec.fit_transform(train_text)

# Create the test matrix following the same approach as in the case of the training matrix
X_test = vec.transform(test_text)

print("Train matrix dimensionality: ", X_train.shape)
print("Test matrix dimensionality: ", X_test.shape)

# Use logistic regression to classify the products of the test set
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict_proba(X_test)

# Write predictions to a file
with open('sample_submission.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    lst = list()
    for i in range(16):
        lst.append('class'+str(i))
    lst.insert(0, "product")
    writer.writerow(lst)
    for i,test_data in enumerate(test_data):
        lst = y_pred[i,:].round(decimals=4).tolist()
        lst.insert(0, test_data)
        writer.writerow(lst)