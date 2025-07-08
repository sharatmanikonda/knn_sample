import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

# Load the dataset
file_path = r"D:\Instilit\Trainings\material_PPTs\DS\13.KNN\Chronic_Kidney_Disease_data.csv"
data = pd.read_csv(file_path)

# Drop non-predictive columns
data = data.drop(columns=["PatientID", "DoctorInCharge"], axis=1)

data.info()

# Handle categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns


# Separate features (X) and target (y)
X = data.drop(columns=["Diagnosis"], axis=1)  # Input features
y = data["Diagnosis"]  # Target variable

# Target variable class proportions - Imbalanced dataset
y.value_counts(normalize=True)


# Split the dataset into training and testing sets - stratified sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)


# Verify the class proportions post stratified sampling
y_train.value_counts(normalize=True)
y_test.value_counts(normalize=True)
y_test.value_counts()


# Scale the feature values
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train KNN Classifier
knn = KNeighborsClassifier(n_neighbors=5)  # k=5 (default)
knn.fit(X_train, y_train)

# Train data predictions
y_pred_train = knn.predict(X_train)

confusion_matrix(y_train, y_pred_train)
accuracy_score(y_train, y_pred_train)


# Make predictions
y_pred = knn.predict(X_test)

#  Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

np.mean(knn.predict(X_test) == y_test)


confusion_matrix(y_test, y_pred)

pd.crosstab(y_test, y_pred, rownames = ['Actual'], colnames = ['Predictions']) 


# Hyperparameter Tunning
acc = []

# Verify the model for a range of 'k' values
for i in range(1, 50, 2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, y_train)
    train_acc = np.mean(neigh.predict(X_train) == y_train)
    test_acc = np.mean(neigh.predict(X_test) == y_test)
    diff = train_acc - test_acc
    acc.append([i, diff, train_acc, test_acc])

# Training accuracies, and Test accuracies for each k value
acc

# Plotting the training and test accuracies for different values of k
# Red circles - training accuracies
plt.plot(np.arange(1, 50, 2), [i[2] for i in acc], "ro-")
# Blue circles - test accuracies
plt.plot(np.arange(1, 50, 2), [i[3] for i in acc], "bo-")


# Best Model
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
confusion_matrix(y_test, y_pred)
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Save model
with open('kNN_model.pkl', 'wb') as f:
    pickle.dump(knn, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
