import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv('model/F_Employee_Attrition_Prediction.csv')

# Define features & target
feature_columns = [
    'Age', 'Gender', 'Years at Company', 'Monthly Income', 'Work-Life Balance',
    'Job Satisfaction', 'Performance Rating', 'Number of Promotions', 'Overtime',
    'Distance from Home', 'Education Level', 'Marital Status', 'Number of Dependents',
    'Job Level', 'Company Size', 'Company Tenure (In Months)', 'Remote Work',
    'Company Reputation', 'Employee Recognition'
]
X = data[feature_columns]
y = data['Attrition']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Evaluate model on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save trained model
with open('model/employee_attrition_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved successfully!")