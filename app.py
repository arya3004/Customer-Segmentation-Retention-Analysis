import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from src.models import predict_single_customer

# Page Configuration
st.set_page_config(
    page_title="Customer Segmentation & Retention Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (sleek premium dashboard look)
st.markdown("""
<style>
    .reportview-container {
        background: #0F172A;
    }
    .main {
        background-color: #0F172A;
        color: #F8FAFC;
    }
    .stMetric {
        background-color: #1E293B;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    div[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }
    h1, h2, h3 {
        color: #38BDF8 !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #0284C7;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #38BDF8;
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# Path definitions
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "segmented_customers.csv")

@st.cache_data
def load_segmented_data():
    if os.path.exists(PROCESSED_DATA_PATH):
        return pd.read_csv(PROCESSED_DATA_PATH)
    return None

@st.cache_resource
def load_model_metadata():
    meta_path = os.path.join(MODEL_DIR, "model_metadata.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    return None

# App header
st.title("📊 Customer Segmentation & Retention Engine")
st.markdown("##### Real-time Customer Churn Prediction and RFM Behavioral Segmentation")

# Check if model has been trained
if not os.path.exists(os.path.join(MODEL_DIR, "best_churn_model.pkl")):
    st.warning("⚠️ Machine Learning Models are not trained yet! Please run the pipeline script first to generate clustering and classification files.")
    if st.button("🚀 Run Data & ML Pipeline"):
        with st.spinner("Executing pipeline (Downloading data, engineering features, fitting clusters, training models)..."):
            import run_pipeline
            run_pipeline.main()
            st.success("Pipeline executed successfully! Reloading...")
            st.rerun()
else:
    df = load_segmented_data()
    meta = load_model_metadata()
    
    if df is not None:
        # KPI Metrics
        total_customers = len(df)
        churn_rate = (df['Churn'] == 1).mean() * 100
        avg_tenure = df['tenure'].mean()
        avg_monthly = df['MonthlyCharges'].mean()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers Analyzed", f"{total_customers:,}", help="Total customers in dataset")
        with col2:
            st.metric("Overall Churn Rate", f"{churn_rate:.2f}%", help="Percentage of customers who left", delta=None)
        with col3:
            st.metric("Average Relationship (Tenure)", f"{avg_tenure:.1f} months", help="Average tenure of customers")
        with col4:
            st.metric("Average Monthly Charges", f"${avg_monthly:.2f}", help="Average monthly billing amount")
            
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 Customer Segmentation (RFM)", 
            "🔮 Churn Risk Calculator (Simulator)", 
            "📈 ML Model Diagnostics",
            "🏢 Actionable Business Playbook"
        ])
        
        with tab1:
            st.subheader("Customer Personas (K-Means Clustering)")
            st.markdown("We segmented customers based on their **Recency/Loyalty** (Tenure), **Frequency/Breadth** (Number of Services), and **Monetary Value** (Monthly Charges).")
            
            # Persona sizes
            persona_counts = df['CustomerPersona'].value_counts().reset_index()
            persona_counts.columns = ['Persona', 'Count']
            
            p_col1, p_col2 = st.columns([2, 3])
            
            with p_col1:
                # Table summary
                summary_df = df.groupby('CustomerPersona').agg(
                    Size=('CustomerPersona', 'count'),
                    AvgTenure=('tenure', 'mean'),
                    AvgServices=('NumServices', 'mean'),
                    AvgMonthlyBilling=('MonthlyCharges', 'mean'),
                    ChurnRate=('Churn', lambda x: f"{(x.mean()*100):.1f}%")
                ).reset_index()
                st.markdown("#### Segment Profiles")
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Pie chart of sizes
                fig_pie = px.pie(
                    persona_counts, values='Count', names='Persona', 
                    title='Distribution of Customer Personas',
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#F8FAFC')
                st.plotly_chart(fig_pie, use_container_width=True)
                
            with p_col2:
                st.markdown("#### 3D RFM Cluster Visualization")
                # 3D scatter plot of RFM
                fig_3d = px.scatter_3d(
                    df.sample(min(len(df), 2000), random_state=42), 
                    x='tenure', y='NumServices', z='MonthlyCharges',
                    color='CustomerPersona', 
                    labels={'tenure': 'Tenure (Months)', 'NumServices': 'Services Count', 'MonthlyCharges': 'Monthly Spend ($)'},
                    title='3D Customer Segments (Sample of 2,000 customers)',
                    color_discrete_sequence=px.colors.qualitative.Safe
                )
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_backgroundcolor="rgba(15, 23, 42, 0.5)",
                        yaxis_backgroundcolor="rgba(15, 23, 42, 0.5)",
                        zaxis_backgroundcolor="rgba(15, 23, 42, 0.5)"
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#F8FAFC',
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
        with tab2:
            st.subheader("Interactive Customer Churn Simulator")
            st.markdown("Adjust customer characteristics below to predict churn probability in real time using the trained **{}** model.".format(meta.get("model_name", "ML Model") if meta else "XGBoost"))
            
            # Predictor inputs split in columns
            sim_col1, sim_col2, sim_col3 = st.columns(3)
            
            with sim_col1:
                st.markdown("#### Demographics & Tenure")
                gender = st.selectbox("Gender", ["Male", "Female"])
                senior_citizen = st.selectbox("Senior Citizen (Age 65+)", [0, 1])
                partner = st.selectbox("Has Partner?", ["Yes", "No"])
                dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
                tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
                
            with sim_col2:
                st.markdown("#### Subscriptions & Billing")
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Paperless Billing?", ["Yes", "No"])
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", 
                    "Mailed check", 
                    "Bank transfer (automatic)", 
                    "Credit card (automatic)"
                ])
                monthly_charges = st.slider("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=70.0)
                # Compute logical total charges
                default_total = float(monthly_charges * tenure)
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=default_total)
                
            with sim_col3:
                st.markdown("#### Core & Add-on Services")
                phone_service = st.selectbox("Phone Service?", ["Yes", "No"])
                multiple_lines = st.selectbox("Multiple Lines?", ["Yes", "No", "No phone service"])
                internet_service = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
                
                # If internet service, show add-ons, else prefill
                if internet_service != "No":
                    online_security = st.selectbox("Online Security Add-on", ["Yes", "No"])
                    online_backup = st.selectbox("Online Backup Add-on", ["Yes", "No"])
                    device_protection = st.selectbox("Device Protection Add-on", ["Yes", "No"])
                    tech_support = st.selectbox("Tech Support Add-on", ["Yes", "No"])
                    streaming_tv = st.selectbox("Streaming TV Service", ["Yes", "No"])
                    streaming_movies = st.selectbox("Streaming Movies Service", ["Yes", "No"])
                else:
                    online_security = "No internet service"
                    online_backup = "No internet service"
                    device_protection = "No internet service"
                    tech_support = "No internet service"
                    streaming_tv = "No internet service"
                    streaming_movies = "No internet service"
                    
            st.markdown("---")
            
            # Prediction trigger
            customer_dict = {
                "gender": gender,
                "SeniorCitizen": int(senior_citizen),
                "Partner": partner,
                "Dependents": dependents,
                "tenure": int(tenure),
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": float(monthly_charges),
                "TotalCharges": float(total_charges)
            }
            
            calc_col1, calc_col2 = st.columns([2, 3])
            
            with calc_col1:
                st.markdown("#### Churn Risk Assessment")
                if st.button("🔮 Calculate Churn Risk Score"):
                    pred, prob = predict_single_customer(customer_dict)
                    prob_pct = prob * 100
                    
                    # Risk label styling
                    if prob < 0.3:
                        risk_level = "🟢 LOW RISK"
                        risk_color = "#22C55E"
                        reco = "Continue standard engagement. Customer is satisfied and loyal."
                    elif prob < 0.6:
                        risk_level = "🟡 MEDIUM RISK"
                        risk_color = "#EAB308"
                        reco = "Proactively engage. Suggest contract extensions, bundle discounts, or email feedback forms."
                    elif prob < 0.8:
                        risk_level = "🟠 HIGH RISK"
                        risk_color = "#F97316"
                        reco = "Immediate retention action required! Flag to support team. Offer a special promotion, discounts, or service upgrade credits."
                    else:
                        risk_level = "🔴 CRITICAL RISK"
                        risk_color = "#EF4444"
                        reco = "Very high risk of leaving! Trigger a premium retention offer. Arrange a direct support callback to address quality or billing issues."
                        
                    st.markdown(f"### Churn Probability: <span style='color:{risk_color}; font-weight:bold'>{prob_pct:.1f}%</span>", unsafe_allow_html=True)
                    st.markdown(f"### Status: <span style='color:{risk_color}; font-weight:bold'>{risk_level}</span>", unsafe_allow_html=True)
                    
                    # Gauge chart
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = prob_pct,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Probability of Churn %", 'font': {'color': "#F8FAFC", 'size': 18}},
                        gauge = {
                            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#F8FAFC"},
                            'bar': {'color': risk_color},
                            'bgcolor': "#1E293B",
                            'borderwidth': 2,
                            'bordercolor': "#334155",
                            'steps': [
                                {'range': [0, 30], 'color': 'rgba(34, 197, 94, 0.15)'},
                                {'range': [30, 60], 'color': 'rgba(234, 179, 8, 0.15)'},
                                {'range': [60, 80], 'color': 'rgba(249, 115, 22, 0.15)'},
                                {'range': [80, 100], 'color': 'rgba(239, 68, 68, 0.15)'}
                            ]
                        }
                    ))
                    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#F8FAFC', height=250, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    
            with calc_col2:
                st.markdown("#### Tailored Customer Strategy")
                if 'reco' in locals():
                    st.info(f"👉 **Model Suggestion**: {reco}")
                    
                    # Detailed personalized actions based on inputs
                    st.markdown("##### 📌 Key Risk Factors Identified in Simulator:")
                    factors = []
                    if contract == "Month-to-month":
                        factors.append("- **Contract Term**: Month-to-month contract provides no locking mechanism. High volatility risk.")
                    if internet_service == "Fiber optic":
                        factors.append("- **Service Quality**: Fiber optic customers show a higher propensity to churn, indicating potential pricing complaints or stability issues.")
                    if tech_support == "No" and internet_service != "No":
                        factors.append("- **Product Support**: Lack of dedicated Technical Support. Enrolling this customer in free basic support could reduce churn by up to 30%.")
                    if online_security == "No" and internet_service != "No":
                        factors.append("- **Add-on Friction**: Lack of Online Security services makes customer vulnerable to competitors offering full-suite security solutions.")
                    if payment_method == "Electronic check":
                        factors.append("- **Billing Friction**: Electronic check payment requires manual activity monthly. Switching them to automatic payment (credit card/bank) is a key churn deterrent.")
                    if tenure <= 6:
                        factors.append("- **Onboarding Risk**: Customer is in their critical first 6 months. Onboarding support is vital.")
                        
                    if factors:
                        for factor in factors:
                            st.write(factor)
                    else:
                        st.write("🟢 No critical risk factors identified. The customer has a strong contract or healthy support configuration.")
                else:
                    st.write("👈 Click 'Calculate Churn Risk Score' to see tailored business recommendations and risk factor breakdowns.")
                    
        with tab3:
            st.subheader("Machine Learning Performance & Diagnosis")
            st.markdown(f"Our models were trained to predict churn. The best model selected was **{meta.get('model_name') if meta else 'XGBoost'}** based on its validation **F1-Score**.")
            
            with open(os.path.join(MODEL_DIR, "all_model_results.pkl"), "rb") as f:
                all_results = pickle.load(f)
                
            # Create metrics dataframe
            m_records = []
            for name, results_dict in all_results.items():
                m_records.append({
                    "Model": name,
                    "Accuracy": f"{results_dict['Accuracy']*100:.2f}%",
                    "Precision": f"{results_dict['Precision']*100:.2f}%",
                    "Recall": f"{results_dict['Recall']*100:.2f}%",
                    "F1-Score": f"{results_dict['F1-Score']:.4f}",
                    "ROC-AUC": f"{results_dict['ROC-AUC']:.4f}"
                })
            st.dataframe(pd.DataFrame(m_records), use_container_width=True, hide_index=True)
            
            diag_col1, diag_col2 = st.columns(2)
            
            with diag_col1:
                st.markdown("#### Understanding the Metrics (Interview Preparation!)")
                st.markdown("""
                * **Recall (Sensitivity)**: *\"Out of all customers who actually churned, how many did we catch?\"*
                  * **Business Value**: In churn, **Recall is the priority**. Missing a churner is expensive. If we miss a customer who was going to leave, we lose their lifetime value.
                * **Precision**: *\"Out of all customers predicted to churn, how many actually churn?\"*
                  * **Business Value**: Protects budget. If we offer $20 credits to 100 people and precision is very low, we are wasting budget on customers who weren't going to leave anyway.
                * **F1-Score**: The harmonic mean of Precision and Recall. Best overall metric for imbalanced datasets.
                * **ROC-AUC**: Represents the model's ability to distinguish between churners and non-churners at all classification thresholds. A score of 0.84 means there is an 84% chance that the model will rank a randomly chosen churner higher than a randomly chosen loyalist.
                """)
                
            with diag_col2:
                st.markdown("#### Feature Importance Profile")
                # Load feature importances if available
                # Let's plot standard feature importances dynamically from the model
                with open(os.path.join(MODEL_DIR, "best_churn_model.pkl"), "rb") as f:
                    best_model = pickle.load(f)
                with open(os.path.join(MODEL_DIR, "model_features.pkl"), "rb") as f:
                    feature_cols = pickle.load(f)
                    
                if hasattr(best_model, 'feature_importances_'):
                    importances = best_model.feature_importances_
                    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(8).reset_index()
                    feat_imp.columns = ['Feature', 'Importance']
                    fig_bar = px.bar(
                        feat_imp, x='Importance', y='Feature', orientation='h',
                        title='Top 8 Drivers of Customer Churn',
                        color='Importance', color_continuous_scale='blues'
                    )
                    fig_bar.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                        font_color='#F8FAFC', yaxis={'categoryorder':'total ascending'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                elif hasattr(best_model, 'coef_'):
                    importances = np.abs(best_model.coef_[0])
                    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False).head(8).reset_index()
                    feat_imp.columns = ['Feature', 'Coefficient']
                    fig_bar = px.bar(
                        feat_imp, x='Coefficient', y='Feature', orientation='h',
                        title='Top 8 Drivers (Logistic Regression Coefficients)',
                        color='Coefficient', color_continuous_scale='blues'
                    )
                    fig_bar.update_layout(
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                        font_color='#F8FAFC', yaxis={'categoryorder':'total ascending'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.write("Feature importance visual not supported for this model class.")
                    
        with tab4:
            st.subheader("Corporate Retention Playbook")
            st.markdown("Here is the operational business roadmap based on our findings:")
            
            st.markdown("""
            ### 🔴 1. Segment: At-Risk Casuals (Churn Risk: ~50-60%)
            * **Who they are**: Newly acquired customers (tenure < 1 year) who join on Month-to-Month contracts. Typically purchase only 1 service (e.g., phone line or basic internet) with no add-ons (Online Backup, Online Security, or Tech Support).
            * **Strategic Priority**: **Onboarding & Relationship Locking**.
            * **Action Plan**:
              * **Incentivize Contract Transition**: Email target offers providing a 15% discount if they move from Month-to-Month to a 1-Year Contract.
              * **Add-on Bundling**: Offer a free 3-month trial of Tech Support or Online Backup. Once a customer installs backups and security, their switching costs rise, reducing churn risk.
              * **Automated Billing Transition**: Switch their payment method from manual "Electronic Check" to automatic payment. Auto-pay customers are 4x less likely to churn.

            ### 🟡 2. Segment: Standard/Mid-Tier (Churn Risk: ~15-20%)
            * **Who they are**: Customers with moderate tenure (1-3 years) and a balanced set of services.
            * **Strategic Priority**: **Customer Lifetime Value (LTV) Maximization & Upselling**.
            * **Action Plan**:
              * **Loyalty Reward Check-ins**: Send a "Thank you for being a customer for X years" email offering early access to streaming upgrades (TV/Movies).
              * **Proactive Service Audits**: Conduct automated diagnostics of connection issues for Fiber Optic users. Fiber optic users complain about pricing and reliability; solving their technical issues before they call support preserves high-margin contracts.

            ### 🟢 3. Segment: Budget Loyalists (Churn Risk: ~5%)
            * **Who they are**: Long-tenure customers (4+ years) who spend very little monthly (typically basic DSL or phone service) and are on 1 or 2-year contracts.
            * **Strategic Priority**: **Low-Cost Maintenance & Goodwill Protection**.
            * **Action Plan**:
              * **Keep Communication Minimal**: Do not upsell them aggressively. They are price-sensitive; aggressive upselling can spark price reviews and lead to churn.
              * **Autorenewal Gratitude**: Automatically roll over their contracts with minor loyalty gifts (e.g., small speed upgrades or price freeze guarantees).

            ### 🔵 4. Segment: High-Value VIPs (Churn Risk: ~10%)
            * **Who they are**: Long-tenure, premium customers (4+ years) utilizing maximum services (Fiber optic, Multiple lines, TV/Movies streaming, full security suite).
            * **Strategic Priority**: **Priority Protection & Loyalty Concierge**.
            * **Action Plan**:
              * **VIP Support Line**: Routing their calls directly to senior tier-2 engineers to minimize hold times and service friction.
              * **Contract Renewal Consultations**: Proactive check-ins 60 days before contract expiry. Offer hardware upgrades (e.g., premium Wi-Fi routers) to cement multi-year commitment.
            """)
    else:
        st.error("Could not load segmented customer data. Make sure run_pipeline.py has executed completely and saved files in the data directory.")
