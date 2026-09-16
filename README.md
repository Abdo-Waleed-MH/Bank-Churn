# Bank Customer Churn Prediction

A machine learning project that predicts whether a bank customer is likely to leave the bank.

The project uses an end-to-end machine learning pipeline with data cleaning, exploratory data analysis, feature engineering, preprocessing, multiple classification models, hyperparameter tuning, and model evaluation.

## Project Objective

The goal is to predict customer churn using customer information.

The target variable is `Exited`.

- `Exited = 0` means the customer stayed
- `Exited = 1` means the customer left

This is a binary classification problem.

## Dataset

The project uses `Churn_Modelling.csv`.

The dataset contains 10,000 customer records and includes features such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card
- Active Member status
- Estimated Salary
- Exited

The columns `RowNumber`, `CustomerId`, and `Surname` are removed because they are not useful for prediction.

## Project Workflow

```text
Load Data
    ↓
Data Exploration
    ↓
Data Cleaning
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Train / Test Split
    ↓
Preprocessing
    ↓
Train Multiple Models
    ↓
Model Evaluation
    ↓
Hyperparameter Tuning
    ↓
Final Model Selection
    ↓
Feature Importance
    ↓
Model Saving
```

## Data Cleaning

The dataset is checked for:

- Missing values
- Duplicate rows
- Irrelevant columns
- Target distribution
- Data types

No missing values or duplicate rows were found in the dataset.

## Exploratory Data Analysis

The EDA was used to understand the relationship between customer features and churn.

Some observations from the project:

- Germany has the highest churn rate among the three countries.
- Female customers have a higher churn rate than male customers.
- Inactive members have a higher churn rate than active members.
- Age shows a strong relationship with churn.
- The classes are not clearly separated using only two features.
- Some outliers appear in `Age`, `CreditScore`, and `NumOfProducts`.

## Feature Engineering

Three new features were created.

### AgeGroup

Customers are grouped into four categories:

- `Young` for age below 30
- `Adult` for age 30 to 44
- `Middle_Aged` for age 45 to 59
- `Senior` for age 60 or older

### IsZeroBalance

A binary feature that indicates whether the customer has a zero account balance.

```python
df_fe['IsZeroBalance'] = (df_fe['Balance'] == 0).astype(int)
```

### BalanceSalaryRatio

This feature compares the customer's balance with their estimated salary.

```python
df_fe['BalanceSalaryRatio'] = np.log1p(
    df_fe['Balance'] / (df_fe['EstimatedSalary'] + 1)
)
```

## Preprocessing

Numerical features are scaled using `StandardScaler`.

Categorical features are encoded using `OneHotEncoder`.

The preprocessing is handled using `ColumnTransformer`.

Numerical features include:

```text
CreditScore
Age
Tenure
Balance
NumOfProducts
HasCrCard
IsActiveMember
EstimatedSalary
IsZeroBalance
BalanceSalaryRatio
```

Categorical features include:

```text
Geography
Gender
AgeGroup
```

## Models

The project compares several classification models:

- Logistic Regression
- Naive Bayes
- K-Nearest Neighbors
- Support Vector Machine
- Random Forest
- XGBoost

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score

F1-Score is used as an important metric because the target classes are imbalanced.

## Hyperparameter Tuning

Hyperparameter tuning was applied to:

- KNN
- Random Forest
- XGBoost

`GridSearchCV` was used to test different parameter combinations.

For the tuned Random Forest, the selected parameters were:

```python
{
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2
}
```

## Model Results

The final model comparison in the notebook produced these test results:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Tuned Random Forest | 0.8385 | 0.5917 | 0.6658 | 0.6266 |
| Tuned XGBoost | 0.7990 | 0.5041 | 0.7641 | 0.6074 |
| XGBoost | 0.7990 | 0.5041 | 0.7641 | 0.6074 |
| SVM | 0.7915 | 0.4918 | 0.7346 | 0.5892 |
| Random Forest | 0.8600 | 0.7848 | 0.4300 | 0.5556 |
| KNN | 0.8440 | 0.6848 | 0.4324 | 0.5301 |
| Naive Bayes | 0.7815 | 0.4709 | 0.5971 | 0.5265 |
| Tuned KNN | 0.8235 | 0.5882 | 0.4423 | 0.5049 |
| Logistic Regression | 0.7270 | 0.3982 | 0.6683 | 0.4991 |

The project selects the **Tuned Random Forest** based on the highest F1-Score in the notebook.

Its test results were:

```text
Accuracy  = 0.8385
Precision = 0.5917
Recall    = 0.6658
F1-Score  = 0.6266
```

## Feature Importance

The tuned Random Forest was also used to examine feature importance.

The highest-ranked features in the notebook were:

| Feature | Importance |
|---|---:|
| Age | 0.2311 |
| NumOfProducts | 0.1897 |
| AgeGroup_Middle_Aged | 0.0975 |
| Balance | 0.0808 |
| EstimatedSalary | 0.0652 |
| CreditScore | 0.0650 |
| BalanceSalaryRatio | 0.0619 |
| IsActiveMember | 0.0512 |
| Geography_Germany | 0.0471 |
| Tenure | 0.0381 |

## Project Structure

```text
Bank-Customer-Churn-Prediction/
│
├── Churn_Modelling.csv
├── Bank_Customer_Churn_Prediction_ML_Pipeline.ipynb
├── bank_churn_model.pkl
├── app.py
├── templates/
│   └── index.html
└── README.md
```

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Joblib
- Jupyter Notebook

## Machine Learning Pipeline

The final deployment package saves both the preprocessing object and the trained model.

```python
deployment_package = {
    "preprocessor": preprocessor,
    "model": best_rf_model
}
```

The saved model can then be loaded and used for prediction on new customer data.

## Web Application

The model can be connected to a simple Flask web application.

The user enters customer information such as:

- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card status
- Active Member status
- Estimated Salary

The application returns:

- Churn prediction
- Churn probability

## Notebook

The main notebook contains the complete analysis and implementation:

`Bank_Customer_Churn_Prediction_ML_Pipeline.ipynb`

It includes data exploration, feature engineering, preprocessing, model comparison, tuning, evaluation, feature importance, and model saving.

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# only needed once, or whenever you want to retrain:
python train_model.py

python app.py
```

Then open **http://127.0.0.1:5000**.

## Authors

Abdulrahman
