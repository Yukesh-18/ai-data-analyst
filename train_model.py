import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

#1. Load data
df = pd.read_csv("data.csv")

#2.Clean it up (this dataset needs TotalCharges converted  to numbers)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

#3. Turn text colomuns into numbers (ML Models only understand numbers)
df_encoded = pd.get_dummies(df.drop('customerID', axis=1), drop_first=True)

#4. Split into features (x) and target (Y) - target = what we're predicting
X = df_encoded.drop('Churn_Yes', axis=1)
Y = df_encoded['Churn_Yes']

#5. Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#6. Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, Y_train)

#7.Check how good it is
prediction = model.predict(X_test)
print("Accuracy:", accuracy_score(Y_test, prediction))
print(classification_report(Y_test, prediction))

#8.Save the model so we can  reuse it later
joblib.dump(model, 'model.pkl')
joblib.dump(X.columns.tolist(), 'columns.pkl')

