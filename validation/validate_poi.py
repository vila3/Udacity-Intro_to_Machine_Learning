#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

### it's all yours from here forward!

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split

np.random.seed(42)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)
print len(labels_test)

tree = DecisionTreeClassifier()
tree.fit(features_train, labels_train)

pred = tree.predict(features_test)
arr = np.zeros(pred.shape[0])

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

print 'Confusion Matrix:\n', confusion_matrix(true_labels, predictions)
print '\nPrecision:', precision_score(true_labels, predictions)
print 'Recall:', recall_score(true_labels, predictions)

print 'Accuracy:', tree.score(features_test, labels_test)
