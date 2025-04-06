import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.mixture import GaussianMixture

# Load data
customer_df = pd.read_csv('customer_data.csv')

# Sidebar inputs for interactivity
st.sidebar.header("Predict Customer Segment")
recency = st.sidebar.number_input("Days since last purchase (Recency)", min_value=0)
frequency = st.sidebar.number_input("Total orders (Frequency)", min_value=1)
monetary = st.sidebar.number_input("Total spend (Monetary)", min_value=0.0)

# Train GMM model
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(customer_df[['Recency', 'Frequency', 'Monetary']])

# Predict cluster for new input
if st.sidebar.button("Predict Segment"):
    cluster = gmm.predict([[recency, frequency, monetary]])[0]
    st.success(f"Predicted Segment: {cluster}")

# ---- Customer Profiler ----
st.sidebar.header("🔍 Customer Profiler")
customer_id = st.sidebar.selectbox(
    "Select CustomerID",
    customer_df['CustomerID'].unique()
)

if st.sidebar.button("Show Profile"):
    profile = customer_df[customer_df['CustomerID'] == customer_id].iloc[0]
    
    st.subheader(f"Profile for Customer {customer_id}")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Segment", f"Cluster {profile['GMM_Cluster']}")
        st.metric("Total Spend", f"${profile['Monetary']:,.2f}")
    
    with col2:
        st.metric("Last Purchase", f"{int(profile['Recency'])} days ago")
        st.metric("Total Orders", int(profile['Frequency']))
    
    # Top Products (if you have this data)
    st.write("**Top Purchased Items**:")
    st.code(f"Item Code: {profile['Top_Item']}")  # Replace with actual product name if available

# ---- What-If Analysis ----
st.header("📊 Business Impact Simulator")

# Inputs for simulation
discount = st.slider("Discount % for At-Risk Customers", 0, 50, 20)
budget = st.number_input("Marketing Budget ($)", 1000, 10000, 5000)

# Simulate ROI
if st.button("Run Simulation"):
    high_value_customers = customer_df[customer_df['GMM_Cluster'] == 2].shape[0]
    potential_sales = high_value_customers * budget * (discount/100)
    st.success(f"**Estimated Revenue Uplift**: ${potential_sales:,.2f} from {high_value_customers} VIP customers!")

# Main visualization
st.title("Customer Segmentation with GMM")
fig = px.scatter(
    customer_df,
    x='Monetary',
    y='Frequency',
    color='GMM_Cluster',
    hover_data=['CustomerID'],
    title="Segments: Monetary vs Frequency"
)
st.plotly_chart(fig)

# ---- Industry Use Cases ----
st.header("🚀 Industry Applications")

# Use Case 1: Personalized Marketing
st.subheader("1. Personalized Marketing Campaigns")
st.markdown("""
- **VIP Customers (Cluster 2)**: Offer exclusive early access to new products.
- **At-Risk Customers (High Recency)**: Send win-back discounts (e.g., *"We miss you! 20% off your next order"*).
- **Frequent Low-Spenders (Cluster 0)**: Bundle deals to increase order value.
""")

# Use Case 2: Inventory Optimization
st.subheader("2. Inventory Optimization")
st.markdown("""
- **Top Products per Segment**: Stock more items frequently bought by high-value clusters.
- **Seasonal Trends**: Adjust inventory before peak buying days (e.g., Black Friday for Cluster 2).
""")

# Use Case 3: Customer Lifetime Value (CLV) Prediction
st.subheader("3. Predict Customer Value")
if st.button("Estimate CLV for Segments"):
    clv = customer_df.groupby('GMM_Cluster')['Monetary'].mean().reset_index()
    clv['CLV (6 months)'] = clv['Monetary'] * 2  # Simplified projection
    st.write(clv)

# Show raw data
if st.checkbox("Show raw data"):
    st.write(customer_df)

# ---- Customer Journey Map ----
st.header("🗺️ Customer Journey by Segment")

# Sample data (replace with your actual metrics)
journey_data = {
    "Segment": ["VIP", "Medium", "Low"],
    "Avg. Purchase Frequency": [15, 5, 2],
    "Avg. Cart Value": ["$200", "$50", "$20"],
    "Preferred Channel": ["Mobile App", "Website", "Email"]
}

st.table(journey_data)



# ---- Export Report (Fixed) ----
from fpdf import FPDF
import base64

def create_report():
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.cell(200, 10, txt="Customer Segmentation Report", ln=True, align='C')
    
    # Key Metrics
    pdf.cell(200, 10, txt=f"Segments: {len(customer_df['GMM_Cluster'].unique())}", ln=True)
    pdf.cell(200, 10, txt=f"Total Customers: {len(customer_df)}", ln=True)
    
    # Cluster Summary Table
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="Cluster Summary:", ln=True)
    clusters = customer_df.groupby('GMM_Cluster').agg({
        'Monetary': 'mean',
        'Frequency': 'mean',
        'Recency': 'mean'
    }).reset_index()
    
    # Add table headers
    pdf.cell(40, 10, txt="Cluster", border=1)
    pdf.cell(40, 10, txt="Avg. Spend", border=1)
    pdf.cell(40, 10, txt="Avg. Orders", border=1)
    pdf.cell(40, 10, txt="Avg. Recency", border=1, ln=True)
    
    # Add table rows
    for _, row in clusters.iterrows():
        pdf.cell(40, 10, txt=str(row['GMM_Cluster']), border=1)
        pdf.cell(40, 10, txt=f"${row['Monetary']:.2f}", border=1)
        pdf.cell(40, 10, txt=str(round(row['Frequency'])), border=1)
        pdf.cell(40, 10, txt=f"{int(row['Recency'])} days", border=1, ln=True)
    
    pdf.output("report.pdf")

if st.button("📥 Generate PDF Report"):
    create_report()
    with open("report.pdf", "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    st.markdown(
        f'<a href="data:application/pdf;base64,{base64_pdf}" download="report.pdf">Download Report</a>',
        unsafe_allow_html=True
    )