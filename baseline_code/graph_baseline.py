import csv
import networkx as nx
import numpy as np

from sklearn.linear_model import LogisticRegression

# Create an undirected graph
G = nx.read_edgelist('edgelist.txt', delimiter=',', create_using=nx.Graph(), nodetype=int)
n = G.number_of_nodes()
m = G.number_of_edges()
print('Number of nodes:', n)
print('Number of edges:', m)

# Read training data
train_data = list()
y_train = list()
with open("y_train.txt", "r") as f:
    for line in f:
        t = line.split(',')
        train_data.append(int(t[0]))
        y_train.append(int(t[1][:-1]))

# Read test data
test_data = list()
with open("test.txt", "r") as f:
    for line in f:
        t = line.split(',')
        test_data.append(int(t[0]))

# Compute nodes' core number
core_num = nx.core_number(G)

# Create the training matrix. Each row corresponds to a node
# Use the following 3 features for each node:
# (1) degree
# (2) core number
# (3) average degree of neighbors
X_train = np.zeros((len(train_data), 3))
avg_neig_deg = nx.average_neighbor_degree(G, nodes=train_data)
for i in range(len(train_data)):
    X_train[i,0] = G.degree(train_data[i])
    X_train[i,1] = core_num[train_data[i]]
    X_train[i,2] = avg_neig_deg[train_data[i]]

# Create the test matrix. Use the same 3 features as above
X_test = np.zeros((len(test_data), 3))
avg_neig_deg = nx.average_neighbor_degree(G, nodes=test_data)
for i in range(len(test_data)):
    X_test[i,0] = G.degree(test_data[i])
    X_test[i,1] = core_num[test_data[i]]
    X_test[i,2] = avg_neig_deg[test_data[i]]

print("Train matrix dimensionality: ", X_train.shape)
print("Test matrix dimensionality: ", X_test.shape)

# Train logistic regression and then use it to make predictions
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