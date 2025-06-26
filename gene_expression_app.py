import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="Gene Expression Analyzer", 
                   page_icon=":dna:", 
                   layout="wide")

# Load data
@st.cache_data
def load_data():
    # Update this to match your actual file format
    df = pd.read_csv('GSE223351_HCC1954_NormalizedCounts.csv')
    genes = df.iloc[:, 0]
    df = df.iloc[:, 1:]  # Remove the first column (gene names)
    df.index = genes     # Set gene names as index
    return df

df = load_data()

# Sidebar controls
st.sidebar.header("Analysis Controls")
selected_samples = st.sidebar.multiselect(
    "Select samples to analyze",
    options=df.columns,
    default=df.columns.tolist()
)

alpha = st.sidebar.slider("Significance threshold (alpha)", 0.001, 0.1, 0.05, 0.001)
fc_threshold = st.sidebar.slider("Fold change threshold (log2FC)", 0.5, 2.0, 1.0, 0.1)
top_n_genes = st.sidebar.slider("Number of top genes to display", 5, 50, 20, 5)

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

# Zero expression analysis
st.header("Zero Expression Analysis")
zero_counts = (df[selected_samples] == 0).sum(axis=0)
fig2 = px.bar(zero_counts, 
             title="Number of Genes with Zero Expression per Sample",
             labels={'value': 'Count of Zero-Expression Genes', 'index': 'Sample'})
st.plotly_chart(fig2, use_container_width=True)

# Expression histogram
st.header("Expression Distribution Histogram")
selected_gene = st.selectbox("Select a gene to view distribution", df.index)
fig3 = px.histogram(df.loc[selected_gene, selected_samples], 
                   title=f"Expression Distribution for {selected_gene} Across Samples",
                   labels={'value': 'Normalized Counts'})
st.plotly_chart(fig3, use_container_width=True)

# Sample Contribution to Total Expression
st.header("Sample Contribution to Total Expression")
sample_totals = df[selected_samples].sum()
fig4 = px.pie(sample_totals, values=sample_totals.values, names=sample_totals.index,
             title="Proportion of Total Expression by Sample")
st.plotly_chart(fig4, use_container_width=True)

# ===== TOP EXPRESSED GENES =====
st.header("🏆 Top 10 Expressed Genes")
top_genes = df[selected_samples].mean(axis=1).sort_values(ascending=False).head(10)

# Create two columns for layout
gene_col1, gene_col2 = st.columns([2, 1])

with gene_col1:
    fig = px.bar(
        top_genes.reset_index(),
        x=top_genes.index,
        y=top_genes.values,
        labels={'x': 'Gene', 'y': 'Average Expression'},
        title="Highest Expressed Genes (Avg Across Samples)",
        color=top_genes.values,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with gene_col2:
    st.subheader("Gene List")
    st.dataframe(
        top_genes.reset_index().rename(columns={'index': 'Gene', 0: 'Expression'}),
        hide_index=True,
        use_container_width=True
    )
    st.download_button(
        "Download Top Genes",
        data=top_genes.to_csv(),
        file_name="top_10_genes.csv",
        mime="text/csv"
    )

# PCA visualization
st.header("Principal Component Analysis")
if len(selected_samples) >= 2:
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
    fig5 = px.scatter(pca_df, x='PC1', y='PC2', text=pca_df.index,
                     title="PCA of Samples",
                     labels={'PC1': f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%})',
                            'PC2': f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%})'})
    fig5.update_traces(textposition='top center', marker=dict(size=12))
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.warning("Need at least 2 samples for PCA")

# TPM Normalization and Differential Expression Analysis
st.header("Differential Expression Analysis")
if len(selected_samples) >= 4:  # Need at least 2 samples per group
    with st.expander("Show Normalization Steps"):
        sizes = df[selected_samples].sum(axis=0)
        tpm = df[selected_samples].div(sizes, axis=1) * 1e6
        tpm_log2 = np.log2(tpm + 0.00001)
        tpm_filtered = tpm_log2[tpm_log2.var(axis=1) > 0]
        
        st.write("Normalized data (log2TPM):", tpm_filtered.head())

    # Define functions for analysis
    def prueba_t(x):
        return ttest_ind(x.iloc[:2], x.iloc[2:]).pvalue

    def fold_change(x):
        return x[:2].mean() - x[2:].mean()

    # Calculate statistics
    p_values = tpm_filtered.apply(prueba_t, axis=1)
    log2FC = tpm_filtered.apply(fold_change, axis=1)
    
    # Create results dataframe
    results = pd.DataFrame({
        "Genes": tpm_filtered.index,
        "p_value": p_values,
        "log2FC": log2FC
    })
    
    # Multiple testing correction
    results["Adj_p"] = multipletests(results["p_value"], method="fdr_bh")[1]
    
    # Classify genes
    results["Clasificacion"] = "NS"
    results.loc[(results["Adj_p"] < alpha) & (results["log2FC"] > fc_threshold), "Clasificacion"] = "Sobreexpresado"
    results.loc[(results["Adj_p"] < alpha) & (results["log2FC"] < -fc_threshold), "Clasificacion"] = "Subexpresado"
    results["neglog10P"] = -np.log10(results["Adj_p"])
    
    st.write("Differential expression results:", results)
    
    # Volcano plot
    st.subheader("Volcano Plot")
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    
    for clase in results["Clasificacion"].unique():
        sub = results[results["Clasificacion"] == clase]
        ax6.scatter(sub["log2FC"], sub["neglog10P"], label=clase)
    
    ax6.axvline(-fc_threshold, linestyle="--", color="gray")
    ax6.axvline(fc_threshold, linestyle="--", color="gray")
    ax6.axhline(-np.log10(alpha), linestyle="--", color="gray")
    
    ax6.set_xlabel("log2FC")
    ax6.set_ylabel("-log10(Adj_p)")
    ax6.legend()
    st.pyplot(fig6)
    
    # Top genes heatmap
    st.subheader(f"Top {top_n_genes} Significant Genes")
    top_genes = results[results["Clasificacion"] != "NS"].sort_values("Adj_p").head(top_n_genes)
    
    if not top_genes.empty:
        # Filter matrix for top genes
        matriz_genes = tpm_log2.loc[top_genes["Genes"]]
        
        # Create heatmap
        fig7, ax7 = plt.subplots(figsize=(12, 8))
        sns.heatmap(matriz_genes, cmap="vlag", ax=ax7)
        ax7.set_title("Expression of Top Significant Genes")
        st.pyplot(fig7)
        
        st.write("Top significant genes:", top_genes)
    else:
        st.warning("No significant genes found with current thresholds")
else:
    st.warning("Need at least 4 samples (2 per group) for differential expression analysis")

# Sample comparison scatter plot
st.header("Sample Comparison")
if len(selected_samples) >= 2:
    sample1 = st.selectbox("Select first sample", selected_samples, index=0)
    sample2 = st.selectbox("Select second sample", selected_samples, index=1)
    
    fig8 = px.scatter(df, x=sample1, y=sample2, 
                     hover_data=[df.index],
                     title=f"Expression Comparison: {sample1} vs {sample2}",
                     labels={'index': 'Gene'})
    fig8.update_traces(marker=dict(size=8, opacity=0.6))
    fig8.add_shape(type="line", x0=0, y0=0, x1=max(df.max()), y1=max(df.max()),
                  line=dict(color="Red", width=2, dash="dot"))
    st.plotly_chart(fig8, use_container_width=True)

# Correlation heatmap
st.header("Sample Correlation Heatmap")
if len(selected_samples) >= 2:
    fig9, ax9 = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[selected_samples].corr(), annot=True, cmap='coolwarm', ax=ax9)
    ax9.set_title("Correlation Between Samples")
    st.pyplot(fig9)
