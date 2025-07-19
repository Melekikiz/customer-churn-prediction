import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, roc_auc_score

df=pd.read_csv("Telco-Customer-Churn.csv")
df.drop('customerID', axis=1, inplace=True)

df['TotalCharges']=pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
df['Churn']=df['Churn'].map({'Yes':1, 'No':0})

categorical_cols=df.select_dtypes(include='object').columns
for col in categorical_cols:
    le=LabelEncoder()
    df[col]=le.fit_transform(df[col])

print("Data info after processing")
print("First 5 roes\n", df.head())
print("\nData info")
print(df.info())

#Faeture-Test Split
X=df.drop('Churn', axis=1)
y=df['Churn']

#Train-Test
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

#Standardization
scaler=StandardScaler()
X_train_scaled =scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

#Model-1
lr=LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr=lr.predict(X_test_scaled)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Classification Report:", classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

#Model-2
rf=RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf=rf.predict(X_test)

print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix", confusion_matrix(y_test, y_pred_rf))

#Feature Importance
feature_importance_df=pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop Important Feature (Random Forest):")
print(feature_importance_df.head(10))

#Visualization
plt.figure(figsize=(14,10))

#Churn
plt.subplot(2,2,1)
sns.countplot(x='Churn', data=df, palette='Set2', legend=False)
plt.title("Churn Distribution")

#Monthly Charges vs Churn
plt.subplot(2,2,2)
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette='Set3')
plt.title("Monthly Charges by Churn")

#Tenure vs Churn
plt.subplot(2,2,3)
sns.boxplot(x='Churn', y='tenure', data=df, palette='coolwarm')
plt.title("Tenure by Churn")

#Feature Importance Plot
plt.subplot(2,2,4)
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis')
plt.title("Top 10 Feature Importances")

plt.tight_layout()
plt.show()

#ROC Curve and AUC for LR
y_prob_lr=lr.predict_proba(X_test_scaled)[:, 1]
fpr_lr, tpr_lr,_=roc_curve(y_test, y_prob_lr)
auc_lr=roc_auc_score(y_test, y_prob_lr)

#ROC Curve and AUC for RF
y_prob_rf=rf.predict_proba(X_test)[:, 1]
fpr_rf, tpr_rf,_=roc_curve(y_test, y_prob_rf)
auc_rf=roc_auc_score(y_test,y_prob_rf)

#Plot ROC
plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC= {auc_lr:.2f})", color='blue')
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.2f})", color='green')
plt.plot([0, 1],[0, 1], linestyle='--', color='gray')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()