import numpy as np
import pandas as pd

class Node:
    def __init__(self, label):
        self.label = label
        self.children = {}
        self.feature_index = 0

class MyDecisionTree:
    def __init__(self, max_depth=None):
        '''
        Initialize the decision tree model

        :param max_depth: maximum depth of the tree
        '''
        self.max_depth = max_depth

    def fit(self, X, y):
        '''
        train the model

        :param X: a 2d array, where each row represents the feature vector of a data point
        :param y: a 1d array, where each element represent the class label of a data point
        :return:
        '''
        self.tree = self._build_tree(X, y)

    def predict(self, X):
        '''
        Predict the class labels for the input data

        :param X: a 2d array, where each row represents the feature vector of a data point
        :return: prediction result
        '''
        prediction = [self._predict(x) for x in X]
        return prediction

    def _entropy(self, y):
        '''
        Calculate entropy

        :param y: a 1d array, where each element represent the class label of a data point
        :return: entropy result
        '''
        _, counts = np.unique(y, return_counts=True)  # Count the number of each class
        probs = counts / np.sum(counts)  # calculate the probability of each class
        return -np.sum(probs * np.log2(probs))

    # def _gini(self, y):
    #     '''
    #     Calculate gini
    #     '''
    #     _, counts = np.unique(y, return_counts=True)  # Count the number of each class
    #     probs = counts / np.sum(counts)  # calculate the probability of each class
    #     return 1 - np.sum(np.square(probs))

    # def _impurity(self, y):
    #     if self.criterion == "entropy":
    #         return self._entropy(y)
    #     elif self.criterion == 'gini':
    #         return self._gini(y)
    #     else:
    #         raise ValueError("invalid criterion")

    def _information_gain(self, X, y, split_attribute_idx):
        '''
        Calculaye the information gain

        :param X: a 2d array, where each row represents the feature vector of a data point
        :param y: a 1d array, where each element represent the class label of a data point
        :param split_attribute_idx: a index of the split attribute
        :return:
        '''
        I_parent = self._entropy(y)
        I_children = 0
        for value in np.unique(X[:, split_attribute_idx]):
            indices = np.where(X[:, split_attribute_idx] == value)[0]
            I_children += len(indices) / len(y) * self._entropy(y[indices])
        return I_parent - I_children


    def _find_best_attribute(self, X, y):
        '''
        Find the best feature to split

        :param X: a 2d array, where each row represents the feature vector of a data point
        :param y: a 1d array, where each element represent the class label of a data point
        :return:
        '''
        best_gain = 0
        best_attribute_idx = None
        for attribute_idx in range(X.shape[1]):
            gain = self._information_gain(X, y, attribute_idx)
            if gain > best_gain:
                best_gain = gain
                best_attribute_idx = attribute_idx
        return best_attribute_idx

    def _build_tree(self, X, y, depth=0):
        '''
        Recursively to build the tree

        :param X: a 2d array, where each row represents the feature vector of a data point
        :param y: a 1d array, where each element represent the class label of a data point
        :param depth: the depth of the current node in the tree
        :return:
        '''
        label = np.bincount(y).argmax()
        node = Node(label=label)
        # if the maximum depth is reached or the current node is turned into a leaf node
        if depth < self.max_depth and len(np.unique(y))>1:
            # choose the best attribute
            best_attribute_idx = self._find_best_attribute(X, y)
            if best_attribute_idx is not None:
                values = np.unique(X[:, best_attribute_idx])
                node.feature_index = best_attribute_idx
                node.children = {}
                for value in values:
                    indices = np.where(X[:, best_attribute_idx] == value)[0]
                    X_subset, y_subset = X[indices], y[indices]
                    subtree = self._build_tree(X_subset, y_subset, depth+1)
                    node.children[value] = subtree
        return node


    def _predict(self, x):
        '''
        Predict the class label for a single sample x

        :param x:
        :return:
        '''
        node = self.tree
        while isinstance(node, Node):
            try:
                node = node.children[x[node.feature_index]]
            except KeyError:
                # if the attribute value is not found in the current node's children, return the current node's label
                return node.label
        return node.label

    def accuracy(self, y_true, y_pred):
        '''
        Calculate the accuracy of the model in test dataset

        :param y_true: a 1d array, true label
        :param y_pred: a 1d array, predict label
        :return:
        '''
        n_correct = sum(1 for i in range(len(y_true)) if y_test[i] == y_pred[i])
        acc = n_correct / len(y_test)
        return acc

    def confusion_matrix(self, y_true, y_pred):
        TP, FN, FP, TN = 0, 0, 0, 0

        for i in range(len(y_true)):
            if y_true[i] == 1 and y_pred[i] == 1:
                TP += 1
            elif y_true[i] == 1 and y_pred[i] == 0:
                FN += 1
            elif y_true[i] == 0 and y_pred[i] == 1:
                FP += 1
            elif y_true[i] == 0 and y_pred[i] == 0:
                TN += 1
        return TP, FN, FP, TN

    def precision(self, y_true, y_pred):
        TP, FN, FP, TN = self.confusion_matrix(y_true, y_pred)
        return TP/(TP+FP)
    def recall(self, y_true, y_pred):
        TP, FN, FP, TN = self.confusion_matrix(y_true, y_pred)
        return TP/(TP+FN)
    def f_score(self, y_true, y_pred):
        TP, FN, FP, TN = self.confusion_matrix(y_true, y_pred)
        return 2*TP/(2*TP+FN+FP)


if __name__ == "__main__":
    from sklearn.metrics import classification_report
    import time
    import psutil
    import os
    import matplotlib.pyplot as plt

    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")

    X = train.iloc[:,:-1].values
    y = train.income.values

    X_test = test.iloc[:,:-1].values
    y_test = test.income.values

    # Get the process ID of the current Python process
    pid = os.getpid()
    # Create a Process object for the current process
    process = psutil.Process(pid)

    performance = {}
    for depth in [i for i in range(2, 25)]:
        start_time = time.time()

        model = MyDecisionTree(depth)
        model.fit(X, y)

        end_time = time.time()

        y_pred = model.predict(X_test)
        acc = model.accuracy(y_test, y_pred)
        exec_time = end_time-start_time
        print(f"Depth: {depth}\n"
              f"Execution time: {exec_time}\n"
              f"Accuracy: {acc}\n")

        performance[depth] = acc, exec_time

    # Get the CPU usage of the process
    cpu_percent = process.cpu_percent()
    # Get the memory usage of the process
    memory_info = process.memory_info()

    sortPerformace = sorted(performance.items(), key=lambda x:x[1][0], reverse=True)
    best_depth = sortPerformace[0][0]
    best_acc = sortPerformace[0][1][0]
    print(f'Best performance with depth {best_depth}, with {best_acc} in accuracy.')
    print(f'CPU usage: {cpu_percent}')
    print(f'Memory usage: {memory_info.rss}\n')

    best_model = MyDecisionTree(best_depth)
    best_model.fit(X, y)
    y_pred = best_model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # draw plot
    speed = []
    accuracy = []
    for key in performance.keys():
        accuracy.append(performance[key][0])
        speed.append(performance[key][1])

    fig, ax1 = plt.subplots()
    line1, = ax1.plot(performance.keys(), speed, color='blue', label='speed')
    p1 = ax1.scatter(performance.keys(), speed, color='blue', label='speed')
    ax1.set_ylabel('Execution time')

    # Create a second y-axis object that shares the same x-axis
    ax2 = ax1.twinx()
    line2, = ax2.plot(performance.keys(), accuracy, color='red', label='accuracy')
    p2 = ax2.scatter(performance.keys(), accuracy, color='red', label='accuracy')
    ax2.set_ylabel('Accuracy')

    ax1.set_xlabel("Depth")
    plt.legend(handles=[p1, p2])
    plt.show()

    # print(f'accuracy = {model.accuracy(y_test, y_pred):.4f}, \n'
    #       f'precision = {model.precision(y_test, y_pred): .4f}, \n'
    #       f'recall = {model.recall(y_test, y_pred): .4f}, \n'
    #       f'f_score = {model.f_score(y_test, y_pred): .4f}')
