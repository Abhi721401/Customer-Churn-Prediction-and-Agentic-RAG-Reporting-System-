# Customer-Churn-Prediction-and-Agentic-RAG-Reporting-System-

IN Progress 

Docker image: https://hub.docker.com/repository/docker/abhijnandas/churn-api/tags

A full end-to-end churn prediction pipeline designed to handle highly imbalanced data using Variational Autoencoders (VAE), traditional ML models, and an Agentic RAG system for automated report generation and explainability.

Project Overview
The objective of this project is to predict customer churn (Yes/No) from structured customer data.
The dataset was highly imbalanced, with the target distribution:

Churn = 0 → 13187 samples
Churn = 1 → 1419 samples

To tackle this, we built a hybrid approach combining unsupervised learning (VAE) and supervised machine learning models, followed by model deployment using FastAPI + Docker.

Key Features
✔ Handles extreme class imbalance using VAE
✔ Two-stage training pipeline: anomaly detection + classification
✔ Multiple ML models trained and evaluated
✔ SHAP-based model explainability
✔ Dockerized API for deployment
✔ Upcoming Agentic RAG pipeline for automated insights & reporting

Project Pipeline
1)  Data Preprocessing
Cleaning & missing value handling
Feature encoding and scaling
Train–test split maintaining class distribution

2) Variational Autoencoder for Imbalance Handling
   
We implemented two VAE-based strategies:

📌 Approach 1: Anomaly Detection using VAE
Train VAE on the majority class (Churn = 0)
Compute reconstruction loss for all samples
Samples with high reconstruction loss → predicted as churn
This improved recall on minority class

📌 Approach 2: Minority Class Data Augmentation
Train a VAE only on churn samples (Churn = 1)
Generate synthetic minority samples
Combine with original dataset
Train ML models on balanced dataset
Result: Better accuracy, precision, recall, and F1-score

3) Supervised Machine Learning Models
We trained and evaluated the following models:
Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
Performance metrics tracked:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
These models showed significant improvement after VAE augmentation.

4) Model Deployment (FastAPI + Docker)
We built an API service that:
✔ Accepts customer features as JSON
✔ Returns churn prediction (0/1)
✔ Returns model confidence
✔ Includes API routes for health check

5)  Explainability using SHAP (Upcoming Section)
We will integrate SHAP to:
Explain individual predictions
Visualize feature importance
Understand what drives churn
This will help business teams take actionable decisions.

6) 🤖  Agentic RAG (Planned)
We are implementing an Agentic Retrieval-Augmented Generation system that will:
Read model outputs
Analyze SHAP explanations
Automatically generate a report
Suggest which customer segments are at risk
Recommend strategies to reduce churn
This will make the system fully self-analyzing and insight-driven.

Results Summary

After applying VAE-based augmentation and anomaly detection:
Accuracy improved
Recall improved significantly (critical for churn)
Precision improved
F1-score improved
The hybrid VAE + ML workflow proved highly effective for imbalanced churn prediction.
