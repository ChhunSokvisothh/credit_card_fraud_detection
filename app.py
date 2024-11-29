import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from io import BytesIO
import joblib
import xgboost as xgb
from catboost import CatBoostClassifier

# Streamlit Config
st.set_page_config(layout="wide", page_title="Credit Card Fraud Detection")

@st.cache_resource
def load_models():
    xgboost_model = xgb.Booster()
    catboost_model = CatBoostClassifier()
    models = {}

    try:
        catboost_model.load_model("model/catboost_model.cbm")
        models["Cat Boost"] = catboost_model
    except FileNotFoundError:
        st.warning("Cat Boost model not found.")

    try:
        xgboost_model.load_model("model/xgboost_model.json")
        models["XG Boost"] = xgboost_model
    except FileNotFoundError:
        st.warning("XG Boost model not found.")

    try:
        models["Random Forest"] = joblib.load("model/random_forest_model.joblib")
    except FileNotFoundError:
        st.warning("Random Forest model not found.")

    try:
        models["Light GBM"] = joblib.load("model/lightgbm_model.pkl")
    except FileNotFoundError:
        st.warning("Light GBM model not found.")

    return models

@st.cache_data
def load_sample_datasets():
    try:
        return {
            "Dataset 1": pd.read_csv("datasets/creditcard.csv"),
            "Dataset 2": pd.read_csv("datasets/creditcard1.csv"),
            "Dataset 3": pd.read_csv("datasets/creditcard2.csv"),
        }
    except FileNotFoundError:
        return {
            "Sample Dataset": pd.DataFrame(
                {
                    "Time": np.arange(100),
                    "Amount": np.random.normal(100, 50, 100),
                    "Class": np.random.choice([0, 1], size=100, p=[0.99, 0.01]),
                    **{f"V{i}": np.random.normal(0, 1, 100) for i in range(1, 29)},
                }
            )
        }

def preprocess_data(data, features):
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data

def detect_features(data):
    v_features = [col for col in data.columns if col.startswith("V") and col[1:].isdigit()]
    if not v_features or "Amount" not in data.columns:
        st.error("Missing required features (`V` columns or `Amount`).")
        return None, None
    return v_features, ["Time"] + v_features + ["Amount"]

def save_to_history(file_name, graphs, fraud_transactions, selected_models):
    """Store the history of graphs, fraud transactions, and selected models."""
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append({
        "file_name": file_name,
        "graphs": graphs,
        "fraud_transactions": fraud_transactions,
        "selected_models": selected_models
    })

def fraud_summary_table(data, models, selected_models, features):
    st.subheader("Fraud Detection Summary Table")
    summary = {"Model": [], "Fraud Detected": []}

    for model_name in selected_models:
        model = models.get(model_name)
        if model is None:
            st.warning(f"Model {model_name} is not available.")
            continue

        try:
            if model_name == "XG Boost":
                dmatrix = xgb.DMatrix(data[features])
                predictions = model.predict(dmatrix)
            else:
                predictions = model.predict(data[features])

            fraud_detected = (predictions == 1).sum()
            summary["Model"].append(model_name)
            summary["Fraud Detected"].append(fraud_detected)
        except Exception as e:
            st.error(f"Error predicting with {model_name}: {e}")
            continue

    summary_df = pd.DataFrame(summary)
    st.table(summary_df)
    return summary_df

def fraud_detection_section(data, models, selected_models, features, file_name):
    st.subheader("Fraud Detection Results")

    # Predictions dictionary
    predictions = {}
    fraud_transactions = {}

    # Run predictions for each selected model
    for model_name in selected_models:
        model = models.get(model_name)
        if model is None:
            st.warning(f"Model {model_name} is not available.")
            continue

        try:
            if model_name == "XG Boost":
                dmatrix = xgb.DMatrix(data[features])
                predictions[model_name] = model.predict(dmatrix)
            else:
                predictions[model_name] = model.predict(data[features])

            # Detect fraud transactions
            fraud_indices = np.where(predictions[model_name] == 1)[0]
            fraud_transactions[model_name] = data.iloc[fraud_indices]
        except Exception as e:
            st.error(f"Error predicting with {model_name}: {e}")
            continue

    # Add tabs for summary and detailed results
    tab1, tab2 = st.tabs(["Summary Table", "Detailed Results"])

    with tab1:
        # Display summary table
        fraud_summary_table(data, models, selected_models, features)

    with tab2:
        # Visualize and analyze detailed results
        summarize_and_visualize(data, predictions, selected_models)

    # Save visualizations and fraud transaction data to history
    graphs = data_visualization_section(data)  # Generate data visualizations
    save_to_history(file_name, graphs, fraud_transactions, selected_models)
    
def summarize_and_visualize(data, predictions, selected_models):
    st.write("### Fraud Detection Summary")
    summary = {}
    fraud_transactions = {}

    for model in selected_models:
        preds = predictions[model]
        fraud_indices = np.where(preds == 1)[0]
        fraud_count = len(fraud_indices)
        fraud_percentage = (fraud_count / len(data)) * 100

        fraud_transactions[model] = data.iloc[fraud_indices]
        summary[model] = {"Fraud Count": fraud_count, "Fraud Percentage": fraud_percentage}

    # Display the fraud summary table
    summary_df = pd.DataFrame(summary).T
    st.write(summary_df)

    # Add the detected fraud transactions table to the Detailed Results section
    st.write("### Detected Fraud Transactions by Model")
    for model, fraud_data in fraud_transactions.items():
        st.write(f"#### {model}")
        if not fraud_data.empty:
            st.dataframe(fraud_data)
        else:
            st.write("No fraud transactions detected by this model.")

    # Plotting fraud likelihood based on transaction amount
    st.write("### Fraud Likelihood Based on Transaction Amount")
    data['Amount_Category'] = pd.cut(data['Amount'], bins=[0, 100, 500, 1000, np.inf], 
                                     labels=["Low (<=100)", "Moderate (100-500)", "High (500-1000)", "Very High (>1000)"])
    fraud_analysis = {}
    for model in selected_models:
        data[f"{model}_Prediction"] = predictions[model]
        fraud_analysis[model] = data.groupby('Amount_Category')[f"{model}_Prediction"].mean() * 100

    fraud_prob_df = pd.DataFrame(fraud_analysis)

    # Plot fraud likelihood by transaction amount
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    fraud_prob_df.plot(kind='bar', ax=ax, colormap="viridis", rot=0)
    ax.set_title("Fraud Probability by Transaction Amount", color='white')
    ax.set_ylabel("Fraud Probability (%)", color='white')
    ax.set_xlabel("Transaction Amount Categories", color='white')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    st.pyplot(fig)

    # Add analysis summary and visualization to history
    st.session_state["history"].append({
        "data": data,
        "summary_df": summary_df,
        "fraud_transactions": fraud_transactions,
        "fraud_prob_df": fraud_prob_df,
        "fig": fig,
        "selected_models": selected_models
    })


def data_visualization_section(data):
    st.subheader("Data Visualizations")
    graphs = []  # Initialize an empty list for graphs

    # Visualization 1: Time Density Plot
    class_0 = data.loc[data["Class"] == 0]["Time"]
    class_1 = data.loc[data["Class"] == 1]["Time"]
    hist_data = [class_0, class_1]
    group_labels = ["Not Fraud", "Fraud"]
    fig1 = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
    fig1.update_layout(title="Credit Card Transactions Time Density Plot", xaxis_title="Time [s]", 
                       paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white"))
    st.plotly_chart(fig1, key="time_density_plot")
    graphs.append(fig1)

    # Visualization 2: Class Balance
    class_balance = data["Class"].value_counts()
    df_balance = pd.DataFrame({"Class": class_balance.index, "Values": class_balance.values})
    fig2 = go.Figure(go.Bar(x=df_balance["Class"], y=df_balance["Values"], 
                            marker=dict(color="Red"), text=df_balance["Values"]))
    fig2.update_layout(title="Credit Card Fraud Class - Data Imbalance", 
                       xaxis_title="Class (0: Not Fraud, 1: Fraud)", yaxis_title="Number of Transactions", 
                       paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white"))
    st.plotly_chart(fig2, key="class_balance_plot")
    graphs.append(fig2)

    # Visualization 3: Fraudulent Transactions
    fraud = data.loc[data["Class"] == 1]
    fig3 = go.Figure(go.Scatter(x=fraud["Time"], y=fraud["Amount"], mode="markers", 
                                marker=dict(color="rgb(238,23,11)", opacity=0.5)))
    fig3.update_layout(title="Amount of Fraudulent Transactions", xaxis_title="Time [s]", yaxis_title="Amount",
                       paper_bgcolor="black", plot_bgcolor="black", font=dict(color="white"))
    st.plotly_chart(fig3, key="fraudulent_transactions_plot")
    graphs.append(fig3)
    return graphs

def main():
    # Initialize session state keys if not already present
    if "page" not in st.session_state:
        st.session_state["page"] = "Sample Dataset"  # Default page

    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Sidebar navigation
    st.sidebar.title("Navigation")
    if st.sidebar.button("Sample Dataset"):
        st.session_state.page = "Sample Dataset"
    if st.sidebar.button("Fraud Detection"):
        st.session_state.page = "Fraud Detection"
    if st.sidebar.button("History"):
        st.session_state.page = "History"

    # Load datasets and models
    sample_datasets = load_sample_datasets()
    models = load_models()

    # Handle the page rendering
    if st.session_state.page == "Sample Dataset":
        st.title("Sample Dataset")
        dataset_name = st.selectbox("Choose a dataset", sample_datasets.keys())
        st.dataframe(sample_datasets[dataset_name].head())

    elif st.session_state.page == "Fraud Detection":
        st.title("Fraud Detection")
        uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            file_name = uploaded_file.name
        else:
            dataset_name = st.selectbox("Choose a sample dataset", sample_datasets.keys())
            data = sample_datasets[dataset_name]
            file_name = dataset_name

        if data is not None:
            v_features, features = detect_features(data)
            if features:
                selected_models = st.multiselect("Select Models", models.keys())
                if st.button("Analyze"):
                    processed_data = preprocess_data(data, features)

                    # Fraud Detection Results Section (includes visualizations)
                    fraud_detection_section(processed_data, models, selected_models, features, file_name)


    elif st.session_state.page == "History":
        st.title("History")
        
        # Check if history is empty
        if not st.session_state["history"]:
            st.info("No history available.")
        else:
            for record in st.session_state["history"]:
                file_name = record.get("file_name", "Unknown File")  # Default to 'Unknown File' if key is missing
                st.subheader(f"File: {file_name}")

                # Display all graphs in the record
                graphs = record.get("graphs", [])
                for graph in graphs:
                    st.plotly_chart(graph)

                # Display fraudulent transactions for each model
                fraud_transactions = record.get("fraud_transactions", {})
                selected_models = record.get("selected_models", [])
                for model in selected_models:
                    st.write(f"#### Fraudulent Transactions Detected by {model}")
                    fraud_data = fraud_transactions.get(model, pd.DataFrame())
                    if not fraud_data.empty:
                        st.dataframe(fraud_data)
                    else:
                        st.write(f"No fraud transactions detected by {model}.")

if __name__ == "__main__":
    main()
