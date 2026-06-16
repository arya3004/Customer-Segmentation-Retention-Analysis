# 📊 Customer Segmentation & Retention Analysis

[![Python Version](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-%231E90FF.svg?style=flat-square)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FF4B4B.svg?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)

A complete, corporate-grade Data Science portfolio project focusing on **Customer Behavioral Segmentation (RFM)** and **Predictive Churn Modeling** using the Telco Customer Churn dataset.

This repository includes:
1. **Behavioral Segmentation**: Adaptation of traditional RFM (Recency, Frequency, Monetary) to a subscription framework using K-Means clustering to discover 4 distinct customer personas.
2. **Churn Prediction**: Side-by-side training and evaluation of Logistic Regression, Random Forest, and XGBoost models, selecting the top performer.
3. **Interactive Streamlit Dashboard**: A high-fidelity, dark-themed UI featuring 3D cluster visualizations, model performance diagnostics, and an interactive customer churn risk calculator.
4. **Jupyter Notebook**: A step-by-step EDA and modeling walkthrough containing clear corporate interpretations.
5. **Interview Prep Q&A**: A master guide containing the exact interview questions recruiters ask about this project.

---

## 🏗️ Project Architecture & Data Flow

```
                 +-----------------------------+
                 |    Raw Telco Dataset        |
                 |      (7,043 rows)           |
                 +--------------+--------------+
                                |
                                v
                 +-----------------------------+
                 |  src/data_loader.py         |
                 |  - Clean TotalCharges (NaN) |
                 |  - Convert Churn -> (0/1)   |
                 +--------------+--------------+
                                |
                                v
                 +-----------------------------+
                 |  src/features.py            |
                 |  - Count active services    |
                 |  - Feature encoding & scaling|
                 +-------+-------------+-------+
                         |             |
        +----------------+             +-----------------+
        |                                                |
        v                                                v
+-------------------------------+              +-------------------------------+
| src/segmentation.py           |              | src/models.py                 |
| - K-Means Clustering (K=4)    |              | - Train classifiers (LR,RF,XG)|
| - Label Customer Personas     |              | - Imbalance ratio balancing   |
| - Save segmentation scaler    |              | - Metric logging (F1, AUC)    |
+---------------+---------------+              +---------------+---------------+
                |                                              |
                +-----------------------+----------------------+
                                        |
                                        v
                         +-----------------------------+
                         | run_pipeline.py             |
                         | - Run end-to-end extraction |
                         +--------------+--------------+
                                        |
                                        v
                         +-----------------------------+
                         | app.py (Streamlit UI)       |
                         | - Interactive Risk Simulator|
                         | - 3D RFM Cluster Visuals    |
                         | - Strategic Business Actions|
                         +-----------------------------+
```

---

## 🎯 Behavioral Segmentation: RFM Adapted for Subscriptions

Traditional RFM (Recency, Frequency, Monetary) was originally built for transactional e-commerce where users make irregular purchases. For subscription services, transactions are periodic (e.g., monthly billing), requiring an adaptation of the RFM variables:

| RFM Metric | E-Commerce Meaning | Subscription Adaptation | Column Proxies |
| :--- | :--- | :--- | :--- |
| **Recency** | Days since last transaction | **Contract Strength & Longevity** | `tenure` (months) & `Contract` type |
| **Frequency** | Number of purchases made | **Product Breadth & Stickiness** | `NumServices` (Count of active services) |
| **Monetary** | Total spend value | **Monthly & Cumulative Spend** | `MonthlyCharges` & `TotalCharges` |

By running K-Means Clustering ($K=4$) on scaled versions of `tenure`, `NumServices`, and `MonthlyCharges`, we uncover **4 Customer Personas**:

1. **At-Risk Casuals**: Short tenure, Month-to-Month contracts, single service, moderate monthly charges. *Churn rate is extremely high (~50-60%)*.
2. **High-Value VIPs**: Long tenure, multi-service users (TV, phone, back-ups, tech support), Fiber Optic, high spend. *Low churn (~10%)*.
3. **Budget Loyalists**: Long tenure, low monthly spend, low service counts, 1-2 year contracts. *Lowest churn (~5%)*.
4. **Standard/Mid-Tier**: Moderate tenure and charges, mid-tier service bundle. *Moderate churn (~15-20%)*.

---

## 🔮 Churn Prediction ML Models

We compared three baseline classifiers. Because of the imbalanced target variable (~26% churn rate), we optimized models using class weight balancing and evaluated performance using **F1-Score** and **ROC-AUC**:

1. **Logistic Regression**: Serves as our explainable baseline. Coefficients help us measure the log-odds change of churn.
2. **Random Forest**: Captures non-linear feature interactions across customer contract terms.
3. **XGBoost Classifier**: Offers top classification performance by iteratively fitting gradient-boosted trees, handling the class imbalance with negative-to-positive ratio adjustments.

---

## 🚀 Setup & Execution

### Prerequisites
Make sure you have Python (3.9+) installed on your machine.

### Installation
1. Clone or download this project to your directory.
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the End-to-End Pipeline
Execute the orchestrator pipeline to download the dataset, process features, generate clusters, and train/save the models:
```bash
python run_pipeline.py
```

### Launching the Streamlit Web Dashboard
Run the dashboard to interactively profile customers and run the simulator:
```bash
streamlit run app.py
```

---

