import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve

df = pd.read_excel("default of credit card clients.xls", header=1)

#print(df.head())

#print(df.isnull().sum())
# Drop ID column
if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)
# Rename target column
df.rename(columns={'default.payment.next.month': 'default_next_month'}, inplace=True)
# Summary statistics
print(df.describe())
print(df.columns.tolist())
# Plot class distribution
sns.countplot(x='default payment next month', data=df)
plt.title('Default vs Non-Default')
#plt.show()

sns.histplot(df['AGE'], bins=20, kde=True)
plt.title('Age Distribution')
#plt.show()

# Train-test split
X = df.drop('default payment next month', axis=1)
y = df['default payment next month']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

try:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
except AttributeError:
    print("Model does not support probability predictions.")
    try:
        y_pred_proba = model.decision_function(X_test)
    except AttributeError:
        y_pred_proba = [0.5] * len(y_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Conclusion example
print("The logsitic regression model achieved {:.2f}% accuracy.".format(accuracy_score(y_test, y_pred) * 100))
print("Possible improvements: feature scaling, trying decision trees or random forests, parameter tuning.")

#results = pd.DataFrame({
#    'Actual': y_test,
#    'Predicted': y_pred,
#    'Probability': y_pred_proba
#})
#results.to_csv('model_results.csv', index=False)

#fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
#roc_df = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds})

#roc_df.to_csv('roc_data.csv', index=False)
