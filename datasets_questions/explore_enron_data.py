#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "Number of people in Enron dataset:", len(enron_data)

qtd = 0
total = 0
for person in enron_data:
    if enron_data[person]["poi"] == 1:
        total += 1
        if enron_data[person]["total_payments"] == 'NaN':
            qtd += 1

print qtd
print (float(qtd)/total)*100, "%"
