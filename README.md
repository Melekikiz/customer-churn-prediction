# customer-churn-prediction
# Customer Churn Prediction Project

This project analyzes customer churn using machine learning models. The goal is to identify factors influencing customer churn and build predictive models that can classify whether a customer is likely to churn.

---

## 📂 Dataset

- **Name:** Telco Customer Churn
- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Target Variable:** `Churn` (Yes / No)

---

## 🔧 Data Preprocessing

- Removed the `customerID` column.
- Converted `TotalCharges` to numeric, removed missing values.
- Encoded the target `Churn` as binary: Yes → 1, No → 0.
- Applied `LabelEncoder` to categorical variables.
- Scaled features using `StandardScaler`.

---

## 🤖 Machine Learning Models

Two models were trained and evaluated:

### 1. Logistic Regression
- Scaled features were used.
- Accuracy: **~80%**
- AUC Score: **~0.84**

### 2. Random Forest Classifier
- Raw (non-scaled) features were used.
- Accuracy: **~79%**
- AUC Score: **~0.83**

---

## 📈 ROC Curve Comparison

The ROC curve shows model performance comparison:

- Logistic Regression performed slightly better in terms of AUC.
- Both models demonstrate good separation between churned and retained customers.



---

## 🔍 Feature Importance (Top 5)

From Random Forest Classifier:

| Feature             | Importance |
|---------------------|------------|
| tenure              | High       |
| MonthlyCharges      | High       |
| Contract            | Medium     |
| TotalCharges        | Medium     |
| OnlineSecurity      | Medium     |

---

## 📊 Visualizations

- **Churn Distribution**: Shows class imbalance.
- **Monthly Charges & Tenure vs Churn**: Customers with higher monthly charges and shorter tenure tend to churn more.
- **Top 10 Important Features**: Visualized based on feature importance scores.

---

## ✅ Conclusion

- Logistic Regression and Random Forest both achieved strong results.
- Tenure, Monthly Charges, and Contract type were key factors.
- Future steps may include:
  - Hyperparameter tuning
  - Trying other classifiers (XGBoost, SVM)
  - Deploying the model with a UI

---

## 📁 Files

- `churn_analysis.py` → Full Python code
- `README.md` → This documentation
- Dataset: [Link to Kaggle dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## 🙋‍♀️ Developed by

**Melek İkiz**  
Freelance Data Scientist  
🔗 [GitHub Profile](https://github.com/Melekikiz)  
🔗 [Upwork Profile](https://www.upwork.com/freelancers/~01ef30341f2458b499)

