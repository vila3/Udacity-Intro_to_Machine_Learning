#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import math
    import operator
    cleaned_data = []

    ### your code goes here

    for i in range(0, len(ages)):
        square_error = math.pow(predictions[i] - net_worths[i], 2)
        cleaned_data.append((ages[i], net_worths[i], square_error))

    for i in range(0, int(round(len(cleaned_data)*0.1, 0))):
        del cleaned_data[cleaned_data.index(max(cleaned_data, key=operator.itemgetter(2)))]

    return cleaned_data

