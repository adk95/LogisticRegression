import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Read file
data = pd.read_csv('breast-cancer.data', header=None)

# Give column names since they are not in the file
data.columns = ['recurrence', 'age', 'menopause', 'tumor_size', 'inv_nodes', 
                'node_caps', 'deg_malig', 'breast', 'breast_quad', 'irradiat']

# Add the columns dynamically in the read file
label_encoders = {}
for column in data.columns:
    if data[column].dtype == 'object':
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

# Set the target column which is 'recurrence' for this set
X = data.drop('recurrence', axis=1)
y = data['recurrence']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

# Calculate the accuracy and create report of different tests
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
