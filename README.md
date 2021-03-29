# Random Forest
To run the program, go to terminal, cd to where python file is, type following commands

python3 random_forest_iris.py

python3 random_forest_car.py

parameter can be change in line 87 of random_forest_car.py:

rf = RandomForest(tree_size = 1, sample_size = 10, depth = 3) # int(len(train_x)*2/3)

-------------------------------------
class DecisionTree(object)

parameters:

- depth, depth of tree, passed from depth of RandomForest

-------------------------------------
class RandomForest(object)

parameters:

- tree_size, number of tree

- sample_size, number of bootstamp sample

- depth, same as decision tree

Test on 2 datasets:
=

iris:
-
Result:

tree_size = 10, sample_size = 20, depth = 3: 49/50/ 98%

tree_size = 10000, sample_size = 10, depth = 3: 48/50 / 96%

tree_size = 10, sample_size = 10, depth = 3: 35/50 / 70%

------------------------------------------------------
CAR:
-
Result:

tree_size = 1, sample_size = 10, depth = 3: 345/519 / 66.47%

tree_size = 10, sample_size = int(len(train_x)*2/3), depth = 3:  358 / 519 / 68.97%

tree_size = 10, sample_size = int(len(train_x)*2/3), depth = 7:  358 / 519 / 68.97%
