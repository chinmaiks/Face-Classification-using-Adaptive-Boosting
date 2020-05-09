#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[ ]:


import numpy as np
import pandas as pd
import math
import os

import cv2

from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

import matplotlib.pyplot as plt

from skimage.transform import integral_image
from skimage.feature import haar_like_feature, haar_like_feature_coord, draw_haar_like_feature


# In[ ]:


dir_path = "Project/2/"

face_images_dir = "data/face"
non_face_imags_dir = "data/non_face/"
img_format = ".bmp"

face_dim = (16, 16)
output_path = "output/"


# # Data Extraction

# In[ ]:


def get_files(path):
    files = []
    for root, subdirs, images in os.walk(path):
        if images:
            full_path_images = [os.path.join(root, image).replace("\\", "/") for image in images]
            files.extend(full_path_images)
    return files


# In[ ]:


def data_preparation():
    face_files = get_files(dir_path + face_images_dir)
    non_face_files = get_files(dir_path + non_face_imags_dir)
    
    tr_face_data_images = face_files[100:200]
    tr_non_face_data_images = non_face_files[100:200]
    
    te_face_data_images = face_files[:100]
    te_non_face_data_images = non_face_files[:100]
    
    tr_face_data = [cv2.imread(img) for img in tr_face_data_images]
    tr_face_data = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) for img in tr_face_data]
    tr_non_face_data = [cv2.imread(img) for img in tr_non_face_data_images]
    tr_non_face_data = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) for img in tr_non_face_data]
    
    te_face_data = [cv2.imread(img) for img in te_face_data_images]
    te_face_data = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) for img in te_face_data]
    
    te_non_face_data = [cv2.imread(img) for img in te_non_face_data_images]
    te_non_face_data = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) for img in te_non_face_data]

    tr_face_data = np.array([cv2.normalize(i, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,                                           dtype=cv2.CV_32F) for i in tr_face_data])
    tr_non_face_data = np.array([cv2.normalize(i, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,                                               dtype=cv2.CV_32F) for i in tr_non_face_data])
    
    te_face_data = np.array([cv2.normalize(i, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,                                           dtype=cv2.CV_32F) for i in te_face_data])
    te_non_face_data = np.array([cv2.normalize(i, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,                                               dtype=cv2.CV_32F) for i in te_non_face_data])
    
    tr_face_labels = np.array([1]*len(tr_face_data))
    tr_non_face_labels = np.array([0]*len(tr_non_face_data))
    
    te_face_labels = np.array([1]*100)
    te_non_face_labels = np.array([0]*100)
    
    return tr_face_data, tr_non_face_data, tr_face_labels, tr_non_face_labels,te_face_data, te_non_face_data, te_face_labels, te_non_face_labels


# In[ ]:


tr_face_data, tr_non_face_data, tr_face_labels, tr_non_face_labels,te_face_data, te_non_face_data, te_face_labels, te_non_face_labels = data_preparation()
len(tr_face_data), len(tr_non_face_data), len(tr_face_labels), len(tr_non_face_labels), len(te_face_data), len(te_non_face_data), len(te_face_labels), len(te_non_face_labels)


# In[ ]:


tr_data = np.concatenate((tr_face_data, tr_non_face_data))
te_data = np.concatenate((te_face_data, te_non_face_data))


# # Extracting HAAR Features and HAAR Co-ordinates

# In[ ]:


X_train = np.array([haar_like_feature(integral_image(tr_data[i]), width=face_dim[0], height=face_dim[1], r=0, c=0) for i in range(len(tr_data))])
y_train = np.array([+1] * len(tr_face_data) + [-1] * len(tr_non_face_data))


# In[ ]:


X_test = np.array([haar_like_feature(integral_image(te_data[i]), width=face_dim[0], height=face_dim[1], r=0, c=0) for i in range(len(te_data))])
y_test = np.array([+1]*len(te_face_data) + [-1]*len(te_non_face_data))


# In[ ]:


feature_coord, feature_type = haar_like_feature_coord(width=face_dim[0], height=face_dim[1])


# In[ ]:


feature_coord.shape, X_train.shape, y_train.shape, X_test.shape, y_test.shape


# # Plotting HAAR Features

# In[ ]:


def plot_haar_feature(best_feature_coordinates, name='ada_boost'):
    for idx, feature_coordinate in enumerate(best_feature_coordinates):
        image = cv2.normalize(tr_data[0], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image = draw_haar_like_feature(image, 0, 0, face_dim[0], face_dim[1], [best_feature_coordinates[idx]])
        plt.imsave(dir_path + output_path + "{}_best_feature_{}".format(name, idx), image)


# # Weak Classifier

# In[ ]:


class WeakClassifier():
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None


# # Selecting Best Features before Boosting

# In[ ]:


class Before_Boosting():
    def __init__(self):
        pass        
        
    def fit(self, X, y):
        #no of training samples and no of features.
        training_samples = X.shape[0]
        no_of_features = X.shape[1]
        
        #keeping track of no of weak classifiers
        self.weak_clfs = []
        
        for i in range(no_of_features):
            min_err = float("inf")
            weak_clf = WeakClassifier()
            feature = np.expand_dims(X[:, i], axis=1)
            unique_feature = np.unique(feature)

            for j in unique_feature:
                p = 1
                prediction = np.ones(np.shape(y))
                prediction[X[:, i] < j] = -1

                err = sum(y==prediction)/len(prediction)

                if err > 0.5:
                    err = 1 - err
                    p = -1

                if err < min_err:
                    weak_clf.polarity = p
                    weak_clf.threshold = j
                    weak_clf.feature_idx = i
                    weak_clf.min_err = err
                    min_err = err

            # Save classifier
            self.weak_clfs.append(weak_clf)
                
    def best_k_feature_ids(self, k):
        arr = np.array([weak_clf.min_err for weak_clf in self.weak_clfs])
        return np.array([self.weak_clfs[i].feature_idx for i in arr.argsort()[:k]])


# In[ ]:


before_boosting = Before_Boosting()


# In[ ]:


before_boosting.fit(X_train, y_train)


# In[ ]:


k = 10
before_boosting_best_features = before_boosting.best_k_feature_ids(k)


# In[ ]:


before_boosting_best_feature_coordinates = np.array([feature_coord[f] for f in before_boosting_best_features])


# In[ ]:


plot_haar_feature(before_boosting_best_feature_coordinates, "before_boosting")


# # Ada Boost

# In[ ]:


class AdaBoost():
    def __init__(self, t=1000):
        self.maxT = t        
        
    def fit(self, X, y):
        #no of training samples and no of features.
        training_samples = X.shape[0]
        no_of_features = X.shape[1]
        
        #initialization of weights
        w = np.full(training_samples, (1 / training_samples))
        
        #keeping track of no of weak classifiers
        self.weak_clfs = []
        
        t = 0
        for t in range(self.maxT):
            print("***************Start t={}***************".format(t))
            weak_clf = WeakClassifier()
            min_err = float("inf")
            
            for i in range(no_of_features):
                feature = np.expand_dims(X[:, i], axis=1)
                unique_feature = np.unique(feature)
                
                for j in unique_feature:
                    p = 1
                    prediction = np.ones(np.shape(y))
                    prediction[X[:, i] < j] = -1
                    
                    err = sum(w[y != prediction])
                    
                    if err > 0.5:
                        err = 1 - err
                        p = -1
                        
                    if err < min_err:
                        weak_clf.polarity = p
                        weak_clf.threshold = j
                        weak_clf.feature_idx = i
                        weak_clf.min_err = err
                        min_err = err
                        
            weak_clf.alpha = 0.5 * math.log((1.0 - min_err) / (min_err))
            predictions = np.ones(np.shape(y))
            negative_idx = (weak_clf.polarity * X[:, weak_clf.feature_idx] < weak_clf.polarity * weak_clf.threshold)
            predictions[negative_idx] = -1

            w *= np.exp(-weak_clf.alpha * y * predictions)
            w /= np.sum(w)

            # Save classifier
            self.weak_clfs.append(weak_clf)
            print("***************End t={}***************".format(t))
                
    def best_k_feature_ids(self, k):
        arr = np.array([weak_clf.min_err for weak_clf in self.weak_clfs])
        return np.array([self.weak_clfs[i].feature_idx for i in arr.argsort()[:k]])
                
    def predict(self, X):
        testing_samples = X.shape[0]
        y_pred = np.zeros((testing_samples, 1))
        for weak_clf in self.weak_clfs:
            predictions = np.ones(np.shape(y_pred))
            negative_idx = (weak_clf.polarity * X[:, weak_clf.feature_idx] < weak_clf.polarity * weak_clf.threshold)
            predictions[negative_idx] = -1
            y_pred += weak_clf.alpha * predictions
        y_pred = np.sign(y_pred).flatten()
        return y_pred


# # Selecting and Plotting Features after performing Ada Boost

# In[ ]:


ada_boost = AdaBoost(100)


# In[ ]:


ada_boost.fit(X_train, y_train)


# In[ ]:


k = 10
best_k_features = ada_boost.best_k_feature_ids(k)


# In[ ]:


best_feature_coordinates = np.array([feature_coord[f] for f in best_k_features])


# In[ ]:


plot_haar_feature(best_feature_coordinates)


# In[ ]:


y_pred = ada_boost.predict(X_test)


# # Accuracy

# In[ ]:


accuracy_score(y_test, y_pred)


# # ROC Curve

# In[ ]:


def plot_roc_curve(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr =  fp/(fp+tn)
    fnr = fn/(tp+fn)
    
    fpr, tpr, threshold = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(dir_path + output_path + "Ada_Boost_roc.png")


# In[ ]:


plot_roc_curve(y_test, y_pred)


# In[ ]:




