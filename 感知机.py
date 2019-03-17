# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 15:16:53 2019

@author: lenovo
"""

from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

X, y = make_blobs(n_samples=1000, centers=2)
y[np.where(y==0)]=-1
x_train, x_test, y_train, y_test = train_test_split(X, y)
x_train = np.c_[x_train, np.ones(x_train.shape[0])]
x_test = np.c_[x_test, np.ones(x_test.shape[0])]

class Perception:
    def __init__(self):
        self.W = None
    def Heaviside_Step(self, z):
        return np.where(z>=0, 1, -1)
    def fit(self, x_train, y_train, epoch, learning_rate):
        n_samples, n_feature = x_train.shape
        self.W = np.random.randn(n_feature)
        losses=[]
        for i in range(epoch):
            y_pred = x_train.dot(self.W)
            loss = sum(np.maximum(0, -y_train * y_pred)) / n_samples
            losses.append(loss)
            if i % 1 == 0:
                print(f"After {i} epoch, loss is {loss}.")
            idx = np.argmax(np.maximum(0, -y_train*y_pred))
            if y_train[idx]*y_pred[idx] > 0:
                break
            dw = x_train[idx]*y_train[idx]
            self.W += learning_rate*dw
        return self.W, losses
    def predition(self, x_test, y_test):
        y_ = self.Heaviside_Step(x_test.dot(self.W))
        score = len(np.where(y_ == y_test)[0])/len(y_test)*100
        print(f"The test data score is {score}%")
        return y_

def plot_hyperplane(X, y, Weight):
    xmin = np.min(X[:, 0])
    xmax = np.max(X[:, 0])
    x_hyperplane = np.array([xmin, xmax])
    y_hyperplane = -(x_hyperplane*Weight[0]+Weight[-1])/Weight[1]
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.plot(x_hyperplane, y_hyperplane, c='b', linewidth=2)
    plt.xlabel('first feature', fontsize=12)
    plt.ylabel('second feature', fontsize=12)
    plt.title("Data and hyperplane", fontsize=16)
    plt.show()

if __name__ == "__main__":
    P = Perception()
    epoch = 1000
    lr = 0.1
    Weight, losses = P.fit(x_train, y_train, epoch, lr)
    y_pred = P.predition(x_test, y_test)
    plot_hyperplane(X, y, Weight)
    plt.show()