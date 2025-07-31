                                           *Loan Status Prediction using Machine Learning*

1) Overview

Loan approval is a critical process in banking and finance, traditionally done through manual evaluation of various applicant details. This project automates that process using **Logistic Regression**, a machine learning algorithm suited for binary classification tasks.

The system predicts whether a loan should be **approved (Y)** or **rejected (N)** based on applicant data such as income, credit history, employment, and property information.

This end-to-end pipeline demonstrates the power of **data preprocessing**, **feature engineering**, **modeling**, **evaluation**, and **visualization** using Python and its powerful libraries.

---

2) Project Goals

- Automate loan approval prediction using a data-driven model.
- Improve decision accuracy and consistency in the financial domain.
- Build a clear and modular machine learning pipeline.
- Generate useful outputs (metrics, graphs, saved model) for interpretation and reuse.

---

3) Dataset Description

The dataset used contains real-world-like information from previous loan applications. Each row represents an applicant and contains:

| Column Name        | Description                                 |
|--------------------|---------------------------------------------|
| Gender             | Male/Female                                 |
| Married            | Marital status                              |
| Dependents         | Number of dependents                        |
| Education          | Graduate/Not Graduate                       |
| Self_Employed      | Employment type                             |
| ApplicantIncome    | Income of the applicant                     |
| CoapplicantIncome  | Income of co-applicant                      |
| LoanAmount         | Requested loan amount                       |
| Loan_Amount_Term   | Duration of the loan                        |
| Credit_History     | Whether they have credit history (0/1)      |
| Property_Area      | Urban/Semiurban/Rural                       |
| Loan_Status        | **Target variable**: Y (approved) or N      |

---

4) Tech Stack and Libraries

| Task                     | Library/Tool         |
|--------------------------|----------------------|
| Data manipulation        | `pandas`, `numpy`    |
| Visualization            | `matplotlib`, `seaborn` |
| Preprocessing            | `LabelEncoder`, `StandardScaler` |
| Modeling                 | `LogisticRegression` |
| Evaluation               | `accuracy_score`, `confusion_matrix`, `classification_report` |
| Model saving             | `joblib`             |

---

5) Workflow and Explanation

### 1. Data Loading and Validation
- The dataset is read using `pandas.read_csv()`.
- If the file is missing, the script raises a `FileNotFoundError`.

### 2. Handling Missing Values
- **Numerical columns**: Filled with median values to prevent skewing.
- **Categorical columns**: Filled with mode (most frequent) to preserve patterns.

### 3. Encoding Categorical Variables
- All string (object) type columns are label encoded.
- The target column `Loan_Status` is also label encoded to convert 'Y'/'N' into `1`/`0`.

### 4. Feature Scaling
- Numerical features are scaled using `StandardScaler` to normalize them around zero, improving logistic regression performance.

### 5. Splitting Data
- Dataset is split into **80% training** and **20% testing** using `train_test_split`.

### 6. Model Training
- A `LogisticRegression` model is trained with `max_iter=1000` to ensure convergence.

### 7. Evaluation Metrics
- Accuracy is printed.
- Classification report includes **precision**, **recall**, and **F1-score**.
- Confusion matrix is plotted and saved.

### 8. Visualization
- **Confusion Matrix**: Visual comparison of actual vs predicted classes.
- **Feature Histograms**: Distribution plots of all numerical features.

### 9. Output Files Generated
- `model_output.txt`: Accuracy and classification report.
- `confusion_matrix.png`: Saved plot of the confusion matrix.
- `feature_histograms.png`: Saved plot of all numerical columns.
- `.pkl` files: Saved trained model and encoders for future use.

---

6) Results

- **Accuracy**: ~81.82%
- The model performs **very well in identifying approved loans (Y)** but has lower performance on rejected ones.
- Confusion matrix and classification report are included for better insight.

---

7) Project Structure

loan-status-prediction/
│
├── loan_status_prediction.py # Main Python script
├── loan_data.csv # Dataset
├── model_output.txt # Text output (accuracy + report)
├── confusion_matrix.png # Evaluation heatmap
├── feature_histograms.png # Feature distributions
├── loan_model.pkl # Saved model
├── scaler.pkl # Scaler object
├── label_encoders.pkl # Encoded feature mappings
├── target_encoder.pkl # Encoded target mappings
└── README.md # Project documentation

8) Authors:
R Harshitha 

Sreeshma K N 

Sushma N 

9) Guided by: Ms. Revathi, Assistant Professor
Department of AI & ML, CMR Institute of Technology

10) License:
This project is developed as part of academic coursework
