Credit Card Default Prediction
==============================



Project Overview
----------------



This project predicts the likelihood of credit card default using machine learning and provides an interactive Tableau dashboard for model evaluation. The goal is to demonstrate the ability to work with real-world financial data, build predictive models, and communicate insights visually.



The dataset used is the Taiwan Credit Card Default dataset, which contains information about credit card clients and whether they defaulted in the following month.

Tools & Technologies
--------------------

-   Python: Data cleaning, feature engineering, and model training
    
-   Scikit-learn: Logistic Regression model and evaluation metrics
    
-   Tableau: Visualization of model performance (Confusion Matrix, ROC Curve, Probability Distribution)
    
-   Pandas / NumPy: Data wrangling and preparation
    
-   Matplotlib / Seaborn: Supplemental visualizations (if needed)
    

Project Workflow
----------------

1.  Data Preparation (Python)
    
    -   Loaded the dataset
        
    -   Cleaned missing values & formatted categorical variables
        
    -   Split into training and test sets
        
    
2.  Model Training (Python)
    
    -   Built a Logistic Regression model
        
    -   Evaluated using Accuracy, Precision, Recall, and ROC-AUC
        
    
3.  Export Results (Python → CSV)
    
    -   Saved predictions (actual_default, pred_default, prob_default) into a CSV file for Tableau
        
    
4.  Visualization (Tableau)
    
    -   Confusion Matrix → shows model classification results with percentages
        
    -   ROC Curve → evaluates the model’s ability to separate defaulters vs. non-defaulters
        
    -   Probability Distribution → compares predicted default probabilities by actual outcome
        
    -   Combined into a Dashboard with KPIs for model performance
        
    

Dashboard Layout
----------------



The Tableau dashboard contains:

-   Title & KPIs (Accuracy, Precision, Recall, Default Rate)
    
-   Confusion Matrix (classification performance)
    
-   ROC Curve (model discriminatory power)
    
-   Probability Distribution (insight into prediction probability spread)
    

Key Insights
------------

-   The model provides a baseline prediction of credit card defaults.
    
-   ROC curve demonstrates predictive power above random guessing.
    
-   Tableau dashboard makes it easy to communicate results with both technical and non-technical audiences.
    

How to Use
----------

1.  Clone this repository.
    
2.  Run python src/credit_card_default.py → outputs saved to outputs/model_results.csv and outputs/roc_data.csv.
    
3.  Open dashboards/Credit_Default_Dashboard.twbx.
    
4.  Load the pre-built Tableau workbook (Credit_Default_Dashboard.twb)
    

Next Steps
----------

-   Experiment with more advanced models (Random Forest, XGBoost).
    
-   Add demographic or financial features (if available) to improve accuracy.
    
-   Deploy as an interactive app (Streamlit / Flask).
