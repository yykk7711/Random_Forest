import random
import pandas as pd
import numpy as np
from decision_tree import *

train_x_file = "car_X_train.csv"
train_x = pd.read_csv("data/" + train_x_file)

train_y_file = "car_y_train.csv"
train_y = pd.read_csv("data/" + train_y_file)

test_x_file = "car_X_test.csv"
test_x = pd.read_csv("data/" + test_x_file)

test_y_file = "car_y_test.csv"
test_y = pd.read_csv("data/" + test_y_file)

reference = ['acc', 'good', 'unacc', 'vgood']

def replacing_index(data):
    feature = data.columns
    count = 0
    for features in list(feature):
        uniq_feature = list(np.unique(data[features]))
        uniq_feature.sort() # ensure the index are the same (also as the reference)
        for row in range(data.shape[0]):
            data.loc[row][list(feature)[count]] = uniq_feature.index((data.loc[row][list(feature)[count]]))
        count += 1

replacing_index(train_x)
replacing_index(train_y)
replacing_index(test_x)
replacing_index(test_y)

train_x = train_x.values.tolist()
train_y = train_y["class"].values.tolist()
test_x = test_x.values.tolist()
test_y = test_y["class"].values.tolist()


class RandomForest(object):
    def __init__(self, tree_size = 20, sample_size = 100, depth = 10, random_state = 2):
        self.tree_size = tree_size
        self.sample_size = sample_size
        self.depth = depth
        random.seed(random_state)
        self.forest = []

    def bootstrap_sample(self, sample_x, sample_y, sample_size):
        x = []
        y = []
        for i in range(sample_size):
            index = random.randrange(len(sample_x)) # with replacement
            x.append(sample_x[index])
            y.append(sample_y[index])
        return x, y

    def fit(self, X, Y):
        for i in range(self.tree_size):
            #print(i)
            dt = DecisionTree()
            x, y = self.bootstrap_sample(X, Y, self.sample_size)
            dt.fit(x, y, self.depth)
            self.forest.append(dt)

    def predict(self, X):
        major = []
        for tree in self.forest:
            major.append(tree.predict(X))
        final = []
        for i in range(len(X)):
            prediction_result = []
            for prediction in major:
                prediction_result.append(prediction[i])
            final.append(self.mode(prediction_result))
        return final

    def mode(self, list):
        max = 0
        unique = np.unique(np.array(list))
        for i in range(len(unique)):
            count = list.count(unique[i])
            if count > max:
                max = count
                m = i
        return unique[m]

rf = RandomForest(tree_size = 1, sample_size = int(len(train_x)*2/3), depth = 3) # int(len(train_x)*2/3)

rf.fit(train_x, train_y)
final = rf.predict(test_x)

# uncommand the following 2 lines to show the final prediction class and it true class
"""
#print(final)
print("Predicted result:")
for unique in np.unique(final):
    count = 0
    for element in range(len(final)):
        if final[element] == unique:
            count += 1
    print("# of %d: %d"%(unique,count))
print("----------------------")
#print(test_y)
print("True result:")
for unique in np.unique(test_y):
    count = 0
    for element in range(len(test_y)):
        if test_y[element] == unique:
            count += 1
    print("# of %d: %d"%(unique,count))
"""

total = len(final)
correct = np.count_nonzero(np.array(final) == np.array(test_y))
print("result: %d / %d"%(correct, total))
