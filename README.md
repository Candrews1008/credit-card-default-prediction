# **Credit Card Default Prediction**

This project predicts whether a credit card customer will default next month using a machine learning approach in Python, with results communicated through an interactive Tableau dashboard.

## **Project Overview**

Dataset: Taiwan Credit Card Default (UCI ML Repository)

Goal: Build a baseline predictive model and visualize its performance with clear, business-friendly dashboards.

Output: A logistic regression model, evaluation metrics, and Tableau visuals including confusion matrix, ROC curve, and probability distributions.

## **Tools & Technologies**

Python: pandas, scikit-learn, matplotlib, seaborn

Tableau: dashboard for evaluation and storytelling

GitHub: reproducible code and stored outputs for Tableau

## **Workflow**

Data Preparation – load, clean, and split the dataset

Model Training – logistic regression baseline

Model Evaluation – accuracy, precision, recall, AUC, confusion matrix, ROC

Export Results – save predictions and ROC curve points to CSV

Visualization in Tableau – import CSVs and build an evaluation dashboard

## **How to Run**

Install dependencies:
pip install -r requirements.txt

Run the Python script:
python src/credit_card_default.py

This will create two CSV files in the outputs/ folder:

 - results_with_features.csv → predictions and probabilities

 - roc_data.csv → FPR/TPR values for ROC curve

Open Tableau and load the CSVs to explore the Credit Default Dashboard.

## **Results**

Accuracy: 80.65%

Precision: 64.23%

Recall: 26.12%

AUC: 0.7045

Interpretation: Accuracy and precision are strong for a first-pass model. AUC of 0.70 shows moderate separation ability. Recall is low, meaning the model misses many true defaults — an area for improvement with threshold tuning or resampling.

## **Dashboard Features**

KPIs: Accuracy, Precision, Recall, AUC

Confusion Matrix: counts and percentages of predictions

ROC Curve: model separation power vs. baseline

Probability Distribution: predicted probabilities by actual default outcome

## **Next Steps**

Adjust classification thresholds to increase recall

Use class_weight="balanced" or resampling techniques for imbalance

Test advanced models (Random Forest, XGBoost)

Add a Tableau parameter for threshold “what-if” analysis

## **Data Source**

UCI Machine Learning Repository — *Default of Credit Card Clients*
