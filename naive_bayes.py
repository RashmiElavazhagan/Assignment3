#-------------------------------------------------------------------------
# AUTHOR: Rashmi Elavazhagan
# FILENAME: naive_bayes.py
# SPECIFICATION: This program will read the weather_training.csv file. Print
# the naive bayes accuracy calculated after all of the predictions.
# FOR: CS 5990- Assignment #3
# TIME SPENT: 60 minutes
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

# Updated classes after discretization
classes_new = [i for i in range(-22, 40, 6)]

# reading the training data
training_data_new = pd.read_csv('weather_training.csv')

# update the training class values according to the discretization (11 values only)
def discretizer_new(data):
    point_ini_new = -110
    for c in classes_new:
        if data["Temperature (C)"] > point_ini_new and data["Temperature (C)"] <= c:
            data["Temperature (C)"] = c
        point_ini_new = c
    return data

discrete_training_new = training_data_new.apply(discretizer_new, axis=1)
y_training_new = np.array(discrete_training_new["Temperature (C)"])
y_training_new = y_training_new.astype(dtype='int')
X_training_new = np.array(discrete_training_new.drop(["Temperature (C)", "Formatted Date"], axis=1).values)

# reading the test data
test_data_new = pd.read_csv('weather_test.csv')

# update the test class values according to the discretization (11 values only)
discrete_test_new = test_data_new.apply(discretizer_new, axis=1)
y_test_new = discrete_test_new["Temperature (C)"]
y_test_new = y_test_new.astype(dtype='int')
X_test_new = discrete_test_new.drop(["Temperature (C)", "Formatted Date"], axis=1).values

# fitting the naive_bayes to the data
clf_new = GaussianNB()
clf_new = clf_new.fit(X_training_new, y_training_new)

# make the naive_bayes prediction for each test sample and start computing its accuracy
# the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
accuracy_new = 0
for (x_testSample, y_testSample) in zip(X_test_new, y_test_new):
    prediction_new = clf_new.predict(np.array([x_testSample]))
    # the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
    # to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    diff_new = 100 * (abs(prediction_new[0] - y_testSample) / y_testSample)
    if diff_new >= -15 and diff_new <= 15:
        accuracy_new += 1

result_new = accuracy_new / len(y_test_new)
print(f"naive_bayes accuracy: {result_new}")
