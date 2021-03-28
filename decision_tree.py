import pandas as pd
import numpy as np


class DecisionTree(object):
    class Node(object):
        def __init__(self, data = None, label = None, depth = 1):
            if data != None:
                self.X = data[0]
                self.Y = data[1]
            self.label = label
            self.best_feature = None
            self.best_value = None
            self.depth = depth

            self.left = None
            self.right = None

    def __init__(self, depth = 3): # at least depth of 2 to have a single layer
        self.depth = depth
        self.root = self.Node()

    def gini(self, Y):
        label = np.unique(Y)
        gini_all = []
        for i in range(len(label)):
            p = Y.count(label[i]) / len(Y)
            Gini = p * (1 - p)
            gini_all.append(Gini)
        return sum(gini_all)

    def gini_of_two_grps(self, Y1, Y2):
        size = len(Y1) + len(Y2)
        return (self.gini(Y1) * len(Y1) + self.gini(Y2)*len(Y2)) / size

    def split(self, feature, value, X, Y):
        # Parameter:
        # feature: int, index of a feature in X

        left_x = []
        left_y = []
        right_x = []
        right_y =[]
        for i in range(len(X)):
            sample = X[i]
            if sample[feature] < value:
                left_x.append(sample)
                left_y.append(Y[i])
            else:
                right_x.append(sample)
                right_y.append(Y[i])
        return left_x, left_y, right_x, right_y

    def best_split(self, X, Y):
        best_gini = 1 # Max value of Gini index
        best_feature = None
        best_value = None
        best_label_left = None
        best_label_right = None
        split = None
        for feature_index in range(len(X[0])): # number of features
            for i in range(len(X)): # number of samples
                left_x, left_y, right_x, right_y = self.split(feature_index, X[i][feature_index], X, Y)
                split = (left_x, left_y, right_x, right_y)
                gini = self.gini_of_two_grps(left_y, right_y)
                # update gini if it perform better
                if gini < best_gini and len(left_x) != 0 and len(right_y) != 0:
                    best_gini = gini
                    best_feature = feature_index
                    best_value = X[i][feature_index]
                    best_label_left = self.mode(left_y)
                    best_label_right = self.mode(right_y)
        return (best_feature, best_value, best_label_left, best_label_right, split)

    def fit(self, X, Y, depth):
        self.root.X = X
        self.root.Y = Y
        self.depth = depth
        self.construct(self.root)

    def construct(self, parent):
        X = parent.X
        Y = parent.Y
        if len(X) == 0 or len(Y) == 0:
            return None
        best_split_current = self.best_split(X, Y)
        left_x = best_split_current[4][0]
        left_y = best_split_current[4][1]
        right_x = best_split_current[4][2]
        right_y = best_split_current[4][3]
        # parent value
        parent.best_feature = best_split_current[0]
        parent.best_value = best_split_current[1]

        best_label_left = best_split_current[2]
        best_label_right = best_split_current[3]

        if len(left_x) == 0:
            parent.left = self.Node((left_x, left_y), best_label_left, self.depth)
        else:
            parent.left = self.Node((left_x, left_y), best_label_left, parent.depth + 1)

        if len(right_x) == 0:
            parent.right = self.Node((right_x, right_y), best_label_right, self.depth)
        else:
            parent.right = self.Node((right_x, right_y), best_label_right, parent.depth + 1)

        if parent.depth + 1 < self.depth:
            self.construct(parent.left)
            self.construct(parent.right)

    def predict(self, X):
        prediction = []
        for i in range(len(X)):
            current = self.root
            while current.depth < self.depth:
                if current.best_feature == None:
                    break
                feature_index = current.best_feature
                value = current.best_value
                if X[i][feature_index] < value:
                    current = current.left
                else:
                    current = current.right
            prediction.append(current.label)
        return prediction

    def mode(self, list):
        max = 0
        unique = np.unique(np.array(list))
        for i in range(len(unique)):
            count = list.count(unique[i])
            if count > max:
                max = count
                m = i
        return unique[m]

if __name__ == "__main__":

    train_x_file = "iris_X_train.csv"
    train_x = pd.read_csv("data/" + train_x_file)

    train_y_file = "iris_y_train.csv"
    train_y = pd.read_csv("data/" + train_y_file)

    test_x_file = "iris_X_test.csv"
    test_x = pd.read_csv("data/" + test_x_file)

    test_y_file = "iris_y_test.csv"
    test_y = pd.read_csv("data/" + test_y_file)

    train_y_list = train_y['Species'].to_list()

    dt = DecisionTree()
    train_x = train_x.values.tolist()
    train_y = train_y.values.tolist()
    test_x = test_x.values.tolist()
    test_y_list = test_y['Species'].to_list()

    depth = 5
    dt.fit(train_x, train_y, depth)

    print(dt.predict(test_x))
    print(test_y_list)

    total = len(test_y_list)
    final = dt.predict(test_x)
    correct = 0
    for i in range(len(final)):
        if final[i] == test_y_list[i]:
            correct += 1

    print("result: %d / %d"%(correct, total))