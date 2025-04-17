import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import plotly.express as px

# Page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# App title
st.title("🧠 Customer Segmentation App")
st.markdown("Upload your RFM dataset and select a clustering algorithm to segment your customers.")

# Step 1: Upload
uploaded_file = st.file_uploader("Upload your processed RFM CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Step 2: Dataset Preview
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Step 3: Feature Selection
    st.subheader("🔧 Select Features for Clustering")
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    features = st.multiselect(
        "Select numeric features (e.g., Recency, Frequency, Monetary)",
        options=numeric_columns,
        default=["Recency", "Frequency", "Monetary"]
    )

    if len(features) >= 2:
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Step 4: Choose Clustering Algorithm
        st.subheader("📌 Choose Clustering Algorithm")
        algo = st.selectbox("Select algorithm", ["KMeans", "DBSCAN", "Agglomerative", "GMM"])

        cluster_labels = None
        silhouette = None

        if algo == "KMeans":
            k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = model.fit_predict(X_scaled)
            silhouette = silhouette_score(X_scaled, cluster_labels)

        elif algo == "DBSCAN":
            eps = st.slider("DBSCAN - eps", 0.1, 5.0, step=0.1, value=0.5)
            min_samples = st.slider("DBSCAN - min_samples", 1, 20, value=5)
            model = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = model.fit_predict(X_scaled)
            if len(set(cluster_labels)) > 1 and -1 not in set(cluster_labels):
                silhouette = silhouette_score(X_scaled, cluster_labels)

        elif algo == "Agglomerative":
            k = st.slider("Select number of clusters", 2, 10, 4)
            model = AgglomerativeClustering(n_clusters=k, linkage="ward")
            cluster_labels = model.fit_predict(X_scaled)
            silhouette = silhouette_score(X_scaled, cluster_labels)

        elif algo == "GMM":
            k = st.slider("Select number of clusters (components)", 2, 10, 3)
            model = GaussianMixture(n_components=k, random_state=42)
            cluster_labels = model.fit_predict(X_scaled)
            silhouette = silhouette_score(X_scaled, cluster_labels)

        # Step 5: Show Results
        st.subheader("📈 Clustering Results")
        df["Cluster"] = cluster_labels

        if silhouette is not None:
            st.success(f"Silhouette Score: {silhouette:.4f}")
        else:
            st.warning("Silhouette Score not applicable (only one cluster or noise present)")

        # Cluster Summary
        st.markdown("### 📋 Cluster Summary")
        cluster_summary = df.groupby("Cluster")[features].mean().round(2)
        cluster_summary["Count"] = df["Cluster"].value_counts().sort_index()
        st.dataframe(cluster_summary)

        # Visualization
        st.markdown("### 📉 Cluster Visualization")
        if len(features) >= 2:
            x_axis = st.selectbox("X-axis", features, index=0)
            y_axis = st.selectbox("Y-axis", features, index=1)
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=df["Cluster"].astype(str),
                hover_data=["Cluster"]
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download segmented data
        st.markdown("### ⬇️ Download Segmented Dataset")
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv, file_name="segmented_customers.csv", mime="text/csv")
    else:
        st.warning("Please select at least 2 features for clustering.")
