# Demand Forecasting & Doctor Visit Prediction with Machine Learning

This project contains two separate machine learning tasks built using real-world style datasets:

1. **Forecasting daily logistics order demand** — a regression problem  
2. **Predicting doctor-visit frequency of older adults** — a classification problem

Both tasks follow a clean ML workflow: data preparation, model training, cross-validation, and performance comparison.

---

## 1. Project Overview

### 1.1 Daily Demand Forecasting (Regression)
The dataset comes from a logistics operation in Brazil and covers 60 days of order activity.  
The goal is to predict the **total number of orders per day** using a set of calendar and operational features (e.g., week of month, day of week, working days, and other operational indicators).

### 1.2 Doctor Visit Prediction (Classification)
The second task uses survey-style data from the **National Poll on Healthy Aging (NPHA)**.  
The aim is to classify older adults into different doctor-visit frequency groups based on demographic and health-related attributes.

---

## 2. Methods and Workflow

For both tasks, the following models were implemented:

- **Linear / Logistic Regression**
- **Support Vector Machine (SVM / SVR)**
- **Decision Tree**
- **Multilayer Perceptron (MLP)** neural network

### General workflow:
1. Load and clean data  
2. Handle missing values  
3. Standardise features where needed  
4. Train models with cross-validation (4–5 folds)  
5. Compare results across models  
6. Summarise findings

---

## 3. Regression Results — Daily Demand Forecasting

Key observations:

- **Decision Tree Regressor**  
  Achieved the **lowest mean squared error**, performing best with depths around 6–8.

- **Support Vector Regression (SVR)**  
  After tuning (e.g., `C=10`, `epsilon=0.1`, `gamma='scale'`), SVR produced strong R² scores and competitive MSE.

- **MLP Regressor**  
  Worked reasonably well, especially with ~50 hidden units. Training required a high iteration limit to converge.

- **Linear Regression**  
  Provided a stable baseline with low variance.

**Summary:**  
Tree-based models and SVR handled the forecasting task most effectively.

---

## 4. Classification Results — Doctor Visit Prediction

Accuracy was the main metric:

- **Logistic Regression**  
  Accuracy around **51%**, with stable validation performance.

- **SVM Classifier**  
  Similar accuracy (~51%) when using a small regularisation parameter (`C=0.01`).

- **Decision Tree Classifier**  
  Achieved about **51.5%** accuracy at `max_depth=5`.

- **MLP Classifier**  
  Lower performance (~43%), but consistent across folds.

**Summary:**  
Logistic Regression and SVM generalised best on this small survey-based dataset.

---

## 5. Repository Structure

```text
.
├── data/
│   ├── daily_demand_orders.csv
│   └── npha_doctor_visits.csv
│
├── notebooks/
│   ├── regression_dt.ipynb
│   ├── regression_svr.ipynb
│   ├── regression_mlp.ipynb
│   ├── classification_logreg.ipynb
│   └── classification_tree_mlp.ipynb
│
├── src/
│   ├── regression_dt.py
│   ├── regression_svr.py
│   └── classification_svm.py
│
├── results/
│
└── README.md
```

---

## 6. How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Launch notebooks
```bash
jupyter notebook
```

Then open any notebook under the `notebooks/` folder.

---

## 7. Key Takeaways

- Different models behave differently depending on task type and dataset characteristics  
- Decision Trees and SVR worked best for the demand forecasting task  
- Logistic Regression and SVM were most reliable for the doctor-visit prediction  
- Cross-validation was essential for understanding model stability  

---

## 8. Possible Improvements

- Add feature engineering for both datasets  
- Try ensemble models (Random Forest, Gradient Boosting, XGBoost)  
- Perform feature importance or SHAP analysis  
- Build a Streamlit demo for interactive prediction  
