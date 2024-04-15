#-------------------------------------------------------------------------
# AUTHOR: Rashmi Elavazhagan
# FILENAME: knn.py
# SPECIFICATION: Complete a KNN classification task with discretized temperature classes
# FOR: CS 5990- Assignment #3
# TIME SPENT: 60 minutes
#-------------------------------------------------------------------------

# importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# defining the hyperparameter values of KNN
k_values_new = [i for i in range(1, 20)]
p_values_new = [1, 2]
w_values_new = ['uniform', 'distance']

# defining the discretization classes
classes_new = [i for i in range(-22, 40, 6)]

# function to discretize the temperature values
def discretize_temperature_new(value):
    for cl in classes_new:
        if value < cl + 3:
            return cl
    return classes_new[-1]

# reading the training data
df_training_new = pd.read_csv('weather_training.csv')
X_training_new = df_training_new.drop(columns=['Formatted Date', 'Temperature (C)']).values
y_training_new = np.array([discretize_temperature_new(x) for x in df_training_new['Temperature (C)'].values]).astype(int)

# reading the test data
df_test_new = pd.read_csv('weather_test.csv')
X_test_new = df_test_new.drop(columns=['Formatted Date', 'Temperature (C)']).values
y_test_real_new = df_test_new['Temperature (C)'].values
y_test_new = np.array([discretize_temperature_new(x) for x in df_test_new['Temperature (C)'].values]).astype(int)

# Normalize the feature data
scaler_new = StandardScaler()
X_training_normalized_new = scaler_new.fit_transform(X_training_new)
X_test_normalized_new = scaler_new.transform(X_test_new)

highest_accuracy_new = 0
best_params_new = {}

# loop over the hyperparameter values (k, p, and w) of KNN
for k in k_values_new:
    for p in p_values_new:
        for w in w_values_new:
            # fitting the KNN to the data
            clf_new = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf_new.fit(X_training_normalized_new, y_training_new)

            # make the KNN prediction for each test sample and start computing its accuracy
            correct_predictions_new = 0
            for x_testSample, y_testRealValue in zip(X_test_normalized_new, y_test_real_new):
                predicted_class_new = clf_new.predict([x_testSample])[0]
                predicted_temperature_new = predicted_class_new
                percentage_difference_new = 100 * abs(predicted_temperature_new - y_testRealValue) / abs(y_testRealValue)

                # the prediction is considered correct if within ±15% of the actual value
                if percentage_difference_new <= 15:
                    correct_predictions_new += 1

            accuracy_new = correct_predictions_new / len(y_test_real_new)

            # check if the calculated accuracy is higher than the previously one calculated
            if accuracy_new > highest_accuracy_new:
                highest_accuracy_new = accuracy_new
                best_params_new = {'k': k, 'p': p, 'weight': w}
                print(f"Highest KNN accuracy so far: {highest_accuracy_new:.2f}")
                print(f"Parameters: k = {k}, p = {p}, weight = {w}")

# After completing the grid search, print out the best parameters and the highest accuracy
print(f"Best parameters: k = {best_params_new['k']}, p = {best_params_new['p']}, weight = {best_params_new['weight']}")
print(f"Highest KNN accuracy: {highest_accuracy_new:.2f}")
