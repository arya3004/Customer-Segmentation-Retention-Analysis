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

## 🎓 Master Interview Preparation Guide

Here are the top 10 questions and high-scoring answers you should review to explain this project in technical interviews:

### Q1: What is customer churn, and why does it matter to a business?
**Answer**: Churn represents the rate at which customers cancel their subscriptions. In subscription businesses, churn directly impacts profitability. It costs roughly 5x more to acquire a new customer than to retain an existing one. High churn creates a "leaky bucket" where customer acquisition costs (CAC) cannot be recovered over the customer's lifetime value (LTV).

---

### Q2: How did you handle the class imbalance in this dataset?
**Answer**: Churn is inherently imbalanced (only ~26% of customers churned). To handle this:
1. **Algorithm Weighting**: We set the `class_weight='balanced'` parameter in Logistic Regression and Random Forest. In XGBoost, we calculated the negative-to-positive class ratio and set `scale_pos_weight` accordingly. This penalizes the model more severely for misclassifying the minority class (churners).
2. **Stratified Splitting**: We used `stratify=y` during `train_test_split` to guarantee that train and validation folds maintain the exact same 26% proportion of churners.
3. **Evaluation Metrics**: We avoided relying on Accuracy and instead evaluated models using **F1-Score** and **ROC-AUC**.

---

### Q3: Why did you choose F1-Score / Recall over Accuracy as your primary model evaluation metric?
**Answer**: If a model simply predicts that *no one* will churn in an environment with 25% churn, it achieves 75% accuracy but is completely useless.
- **Recall (Sensitivity)** measures the percentage of actual churners that the model successfully catches. In churn prevention, **Recall is the priority** because missing a customer who is about to leave means losing their entire future revenue.
- **Precision** measures the proportion of predicted churners who actually churned. A model with low precision will target loyal customers with discount offers, causing unnecessary margin erosion.
- **F1-Score** (harmonic mean of Precision and Recall) allows us to balance these two priorities, which is critical for making cost-efficient business interventions.

---

### Q4: What does a False Positive and False Negative mean in a churn prediction context, and which is more costly?
**Answer**:
- **False Positive (Type I Error)**: The model predicts a customer will churn, but they were actually planning to stay.
  - *Business Cost*: We might waste budget by giving them an unnecessary discount or retention offer, which slightly erodes our margins.
- **False Negative (Type II Error)**: The model predicts a customer will stay, but they actually leave.
  - *Business Cost*: Very high. We lose the customer entirely, along with their Customer Lifetime Value (LTV), and must spend high acquisition costs (CAC) to replace them.
- **Conclusion**: A False Negative is significantly more costly than a False Positive. Therefore, we design our classification threshold to favor high Recall (minimizing False Negatives).

---

### Q5: How did you adapt traditional RFM analysis (originally for transactional e-commerce) to a subscription service?
**Answer**: E-commerce RFM uses Recency (days since purchase), Frequency (number of orders), and Monetary (total order spend). 
In a subscription model, customers are billed on a fixed billing cycle, rendering traditional recency and frequency metrics flat. We adapted the metrics as follows:
- **Recency Proxy**: Customer tenure and contract commitment. Long tenure represents low risk, whereas month-to-month contracts represent an active monthly renewal decision point (high recency risk).
- **Frequency Proxy**: Product breadth/stickiness. We counted the number of active services (e.g., Phone, Fiber Optic, Online Backup, Tech Support). The more services a customer utilizes, the more deeply embedded they are in our ecosystem, leading to higher friction when switching.
- **Monetary Proxy**: MonthlyCharges.

---

### Q6: Explain the K-Means algorithm. How did you choose the number of clusters (K=4), and how did you scale features?
**Answer**: K-Means is a distance-based, unsupervised partitioning algorithm. It groups data points by minimizing the Sum of Squared Errors (SSE) within each cluster.
- **Determining K**: We analyzed the "Elbow Method" (distortion curve) and business feasibility. A model with $K=4$ was optimal because it segmented customers into clean, actionable cohorts (VIPs, Budget Loyalists, Casuals, Mid-tier) that fit distinct marketing channels.
- **Feature Scaling**: K-Means relies on Euclidean distance. If we cluster tenure (scale 0-72 months) and monthly spend (scale 15-120 dollars) without scaling, the monthly spend will dominate the distance calculation due to its larger numerical range. We used `StandardScaler` to normalize the mean to 0 and variance to 1.

---

### Q7: Why is standardization necessary before running K-Means clustering?
**Answer**: K-Means uses Euclidean distance:
$$d(x, y) = \sqrt{\sum (x_i - y_i)^2}$$
If one feature has a scale of 0 to 100,000 (e.g., TotalCharges) and another has a scale of 0 to 5 (e.g., NumServices), a minor change in TotalCharges will completely overwhelm any change in NumServices in the distance calculation. Without standardization, the cluster shapes would be driven entirely by the largest-scale column.

---

### Q8: What are the top features driving customer churn in your model, and what is the business rationale?
**Answer**: The top drivers of churn are:
1. **Contract Type (Month-to-month)**: This is the strongest driver. Month-to-month contracts have zero switching barriers.
2. **Internet Service (Fiber Optic)**: Fiber optic customers are surprisingly high-churn. Further investigation reveals Fiber Optic carries a premium price point; if they experience service issues or price increases, they readily switch.
3. **No Tech Support / No Online Security**: Customers lacking these add-on services show high churn.
4. **Payment Method (Electronic Check)**: Electronic checks require manual action every month, forcing the customer to re-evaluate their purchase decision, unlike automatic billing.

---

### Q9: If a customer is flagged as high-risk, what concrete business strategies should be deployed?
**Answer**: We implement a segmented playbook:
- **For At-Risk Casuals** (short tenure, month-to-month): Run onboarding email campaigns offering a contract-upgrade discount (e.g., switch to a 1-year contract for 15% off) and a free bundle of "Tech Support" or "Online Backup" to increase stickiness.
- **For High-Value VIPs** (high-spend Fiber optic users): Provide priority routing in customer support, run proactive speed/quality audits to address reliability complaints, and offer router upgrades.
- **For Automatic Billing**: Incentivize transition to auto-pay (credit card or bank transfer) by giving a one-time $10 credit.

---

### Q10: How would you deploy this churn model in a production environment and monitor it for data drift?
**Answer**:
1. **Serving**: Package the trained model (`best_churn_model.pkl` and preprocessing scaler) into a REST API using FastAPI. Host it on a containerized service like AWS ECS or Google Cloud Run.
2. **Integration**: The web app or CRM (like Salesforce) calls this API during customer check-ins to flag churn risks in real-time.
3. **Monitoring Drift**: 
   - **Data Drift**: Set up a pipeline to run weekly Kolmogorov-Smirnov tests on numeric features (like `MonthlyCharges`) to detect changes in customer populations.
   - **Concept Drift**: Track the actual churn outcomes against the model's predicted probabilities over time. If accuracy or F1-Score decays, trigger automated model retraining with the latest data.
