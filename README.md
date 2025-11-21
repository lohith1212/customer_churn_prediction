# Customer Churn Prediction (End-to-End Machine Learning Project)

This project predicts whether a telecom customer will **churn** using advanced, production-ready machine learning techniques.  
It includes a full training pipeline, model tuning, handling imbalanced data, and a reusable prediction system.

---

Key Features:

-  Full end-to-end ML pipeline  
-  Feature scaling (StandardScaler)  
-  Label encoding for categorical variables  
-  **SMOTE + Downsampling** for imbalanced dataset handling  
-  **Stratified K-Fold Cross Validation**  
-  Hyperparameter tuning using **RandomizedSearchCV**  
-  Trained & compared 3 ML models:  
  - Logistic Regression  
  - Random Forest  
  - XGBoost  
-  **Automatic best model selection (AUC-based)**  
-  Saves:
  - `best_churn_model.pkl`  
  - `encoders.pkl`  
  - `scaler.pkl`  
-  Production-ready prediction pipeline  
-  Clear modular folder structure  
-  Beginner-friendly & interview-ready project

---

##  Machine Learning Techniques Used

- Logistic Regression  
- Random Forest  
- XGBoost  
- SMOTE (Oversampling)  
- Random Under Sampling (RUS)  
- Feature Standardization  
- Stratified K-Fold  
- RandomizedSearchCV for hyperparameter tuning  
- ROC-AUC as primary evaluation metric  

---

##  How to Run
###  Install dependencies
###  Run the Jupyter Notebook  
Open:

Run all cells to train, tune, and evaluate the models.

### 3️ Use the prediction pipeline
```python
from src.prediction_pipeline import predict_customer

sample_customer = {
    'gender': 'Female',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'tenure': 1,
    'PhoneService': 'No',
    'MultipleLines': 'No phone service',
    'InternetService': 'DSL',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 29.85,
    'TotalCharges': 29.85
}

print(predict_customer(sample_customer))





