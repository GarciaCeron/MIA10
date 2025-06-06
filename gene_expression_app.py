import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Set page config
st.set_page_config(page_title="Gene Expression Analyzer", 
                   page_icon=":dna:", 
                   layout="wide")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('GSE223351_HCC1954_NormalizedCounts.csv')
    df = df.set_index(df.columns[0])
    return df

df = load_data()

# Sidebar controls
st.sidebar.header("Controls")
selected_samples = st.sidebar.multiselect(
    "Select samples to analyze",
    options=df.columns,
    default=df.columns.tolist()
)

top_n_genes = st.sidebar.slider(
    "Number of top genes to display",
    min_value=5,
    max_value=50,
    value=10
)

# Main app
st.title("Gene Expression Data Analysis")
st.markdown("""
This app analyzes normalized gene expression counts from the HCC1954 cell line dataset.
""")

# Data overview
st.header("Data Overview")
st.dataframe(df.describe(), use_container_width=True)

# Sample distribution
st.header("Sample Expression Distribution")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df[selected_samples], ax=ax1)
ax1.set_ylabel("Normalized Counts")
ax1.set_title("Expression Distribution Across Samples")
st.pyplot(fig1)

# Top expressed genes
st.header(f"Top {top_n_genes} Expressed Genes")
top_genes = df.mean(axis=1).sort_values(ascending=False).head(top_n_genes)
fig2 = px.bar(top_genes, x=top_genes.values, y=top_genes.index, 
              orientation='h', 
              title=f"Top {top_n_genes} Highest Expressed Genes (Average Across Samples)",
              labels={'x': 'Average Normalized Counts', 'y': 'Gene'})
st.plotly_chart(fig2, use_container_width=True)

# Sample comparison scatter plot
if len(selected_samples) >= 2:
    st.header("Sample Comparison")
    sample1 = st.selectbox("Select first sample", selected_samples, index=0)
    sample2 = st.selectbox("Select second sample", selected_samples, index=1)
    
    fig3 = px.scatter(df, x=sample1, y=sample2, 
                     hover_data=[df.index],
                     title=f"Expression Comparison: {sample1} vs {sample2}",
                     labels={'index': 'Gene'})
    fig3.update_traces(marker=dict(size=8, opacity=0.6))
    fig3.add_shape(type="line", x0=0, y0=0, x1=max(df.max()), y1=max(df.max()),
                  line=dict(color="Red", width=2, dash="dot"))
    st.plotly_chart(fig3, use_container_width=True)

# Expression histogram
st.header("Expression Distribution Histogram")
selected_gene = st.selectbox("Select a gene to view distribution", df.index)
fig4 = px.histogram(df.loc[selected_gene], 
                   title=f"Expression Distribution for {selected_gene} Across Samples",
                   labels={'value': 'Normalized Counts'})
st.plotly_chart(fig4, use_container_width=True)

# Zero expression analysis
st.header("Zero Expression Analysis")
zero_counts = (df == 0).sum(axis=0)
fig5 = px.bar(zero_counts, 
             title="Number of Genes with Zero Expression per Sample",
             labels={'value': 'Count of Zero-Expression Genes', 'index': 'Sample'})
st.plotly_chart(fig5, use_container_width=True)

# Correlation heatmap
st.header("Sample Correlation Heatmap")
fig6, ax6 = plt.subplots(figsize=(8, 6))
sns.heatmap(df[selected_samples].corr(), annot=True, cmap='coolwarm', ax=ax6)
ax6.set_title("Correlation Between Samples")
st.pyplot(fig6)

# PCA visualization
st.header("Principal Component Analysis")
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Prepare data
X = df[selected_samples].T
X_scaled = StandardScaler().fit_transform(X)

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(data=principal_components, 
                     columns=['PC1', 'PC2'],
                     index=selected_samples)

# Plot
fig7 = px.scatter(pca_df, x='PC1', y='PC2', text=pca_df.index,
                 title="PCA of Samples",
                 labels={'PC1': f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})',
                        'PC2': f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})'})
fig7.update_traces(textposition='top center', marker=dict(size=12))
st.plotly_chart(fig7, use_container_width=True)

# Sample comparison pie chart
st.header("Sample Contribution to Total Expression")
sample_totals = df[selected_samples].sum()
fig8 = px.pie(sample_totals, values=sample_totals.values, names=sample_totals.index,
             title="Proportion of Total Expression by Sample")
st.plotly_chart(fig8, use_container_width=True)
