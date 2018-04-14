#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, PERF_FORMAT_STRING, RESULTS_FORMAT_STRING

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'rate_from_this_person_to_poi', 'rate_from_poi_to_this_person', 'salary', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'long_term_incentive', 'restricted_stock',
                 'director_fees']  # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

for person in my_dataset:
    from_poi = my_dataset[person]['from_this_person_to_poi']
    to_poi = my_dataset[person]['from_poi_to_this_person']
    from_m = my_dataset[person]['from_messages']
    to_m = my_dataset[person]['to_messages']
    if to_poi == 'NaN':
        to_poi = 0.0
    if from_poi == 'NaN':
        from_poi = 0.0
    if from_m == 'NaN':
        my_dataset[person]['rate_from_this_person_to_poi'] = 'NaN'
    else:
        my_dataset[person]['rate_from_this_person_to_poi'] = float(from_poi) / from_m
    if to_m == 'NaN':
        my_dataset[person]['rate_from_poi_to_this_person'] = 'NaN'
    else:
        my_dataset[person]['rate_from_poi_to_this_person'] = float(to_poi) / to_m

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

estimators = [()('pca', PCA(n_components=6)),
              ('kneighbors', KNeighborsClassifier(leaf_size=30, n_neighbors=3, weights='distance'))]
clf = Pipeline(estimators)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
labels_pred = clf.predict(features_test)

true_negatives = 0
false_negatives = 0
false_positives = 0
true_positives = 0
for prediction, truth in zip(labels_pred, labels_test):
    if prediction == 0 and truth == 0:
        true_negatives += 1
    elif prediction == 0 and truth == 1:
        false_negatives += 1
    elif prediction == 1 and truth == 0:
        false_positives += 1
    elif prediction == 1 and truth == 1:
        true_positives += 1
    else:
        print "Warning: Found a predicted label not == 0 or 1."
        print "All predictions should take value 0 or 1."
        print "Evaluating performance for processed predictions:"
        break

try:
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
    precision = 1.0 * true_positives / (true_positives + false_positives)
    recall = 1.0 * true_positives / (true_positives + false_negatives)
    f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
    f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
    print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision=5)
    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives,
                                       true_negatives)
    print ""
except:
    print "Got a divide by zero when trying out:", clf
    print "Precision or recall may be undefined due to a lack of true positive predicitons."
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
