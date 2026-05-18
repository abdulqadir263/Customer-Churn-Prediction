# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to leave the service based on their account and usage data. Built as part of an AI/ML course and designed to cover a complete end-to-end classification pipeline.

## What This Project Does

The goal is simple: given a customer's information, predict if they will churn (leave) or stay. The project walks through every step from raw data to a trained and evaluated model, including data cleaning, feature encoding, scaling, model training, and result comparison.

Two classification models are trained and compared, Logistic Regression and K-Nearest Neighbors, using standard evaluation metrics.

## Dataset

Source: IBM Telco Customer Churn dataset, available on Kaggle.

Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset has around 7,000 rows and 21 columns. Each row represents one customer. The target column is Churn which holds Yes or No values.

## Project Structure

    customer_churn_prediction.py    main script with all steps
    README.md                       this file
    plot1_churn_distribution.png    bar chart of churn counts
    plot2_monthly_charges_boxplot   monthly charges split by churn
    plot3_tenure_histogram.png      tenure distribution by churn
    plot4_correlation_heatmap.png   correlation among numeric features
    plot5_confusion_matrices.png    confusion matrices for both models
    plot6_model_comparison.png      metric comparison bar chart
    plot7_knn_k_selection.png       train vs test accuracy across K values
    plot8_feature_importance.png    top features from logistic regression

## Steps Covered

The script is divided into clear sections so each part can be read and run independently.

Data is loaded and inspected first. The TotalCharges column is stored as a string in the raw file and gets converted to float, with any resulting nulls filled using the median. The customerID column is dropped since it carries no predictive value.

Categorical columns are label encoded and numerical features are scaled using StandardScaler. The scaler is fit only on training data to prevent data leakage.

The data is split 80/20 for training and testing with stratification on the target column to preserve the churn ratio in both splits.

Both models are evaluated using accuracy, precision, recall, F1-score, and confusion matrices. A train vs test accuracy comparison is included to check for overfitting. A K selection plot is also generated to show how KNN performance changes across different values of K.

## Results

Logistic Regression generally performs well on this dataset with stable precision and recall. KNN is more sensitive to the choice of K and benefits from the scaling step. The K selection plot makes it easy to spot where overfitting begins as K decreases toward 1.

## Key Concepts Demonstrated

Classification with imbalanced classes, label encoding, standard scaling, train-test splitting with stratification, overfitting detection by comparing train and test accuracy, and feature importance interpretation through logistic regression coefficients.
