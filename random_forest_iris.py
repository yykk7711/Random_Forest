import random
import pandas as pd
import numpy as np
from decision_tree import *

train_x_file = "iris_X_train.csv"
train_x = pd.read_csv("data/" + train_x_file)

train_y_file = "iris_y_train.csv"
train_y = pd.read_csv("data/" + train_y_file)

test_x_file = "iris_X_test.csv"
test_x = pd.read_csv("data/" + test_x_file)

test_y_file = "iris_y_test.csv"
test_y = pd.read_csv("data/" + test_y_file)


class RandomForest(object):
    def __init__(self, tree_size = 10, sample_size = 20, depth = 3, random_state = 2):
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

rf = RandomForest()

train_x = train_x.values.tolist()
train_y = train_y.values.tolist()
test_x = test_x.values.tolist()
test_y_list = test_y['Species'].to_list()

rf.fit(train_x, train_y)
final = rf.predict(test_x)
print(final)

total = len(test_y_list)
correct = np.count_nonzero(np.array(final) == np.array(test_y_list))
print("result: %d / %d"%(correct, total))