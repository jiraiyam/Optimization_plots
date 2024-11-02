import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from dataclasses import dataclass
from typing import List, Optional, Tuple
from math import pi
from sklearn.manifold import TSNE

from scipy.cluster.hierarchy import dendrogram, linkage


@dataclass
class DashboardConfig:
    """Configuration settings for the dashboard."""
    fig_height: int = 6
    fig_width: int = 10
    style: str = "darkgrid"
    palette: str = "deep"


class DataProcessor:
    """Handles data preprocessing and transformations."""

    @staticmethod
    def process_excel_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process the input Excel data."""
        # Fix column names
        df.columns = [
            df.columns[i - 1] if 'Unnamed' in col else col
            for i, col in enumerate(df.columns)
        ]

        # Create consistent column naming
        df.columns = [
            f'{col}_Values' if i % 2 == 0 else f'{col}_Time'
            for i, col in enumerate(df.columns)
        ]

        # Remove header row and convert to numeric
        df = df.iloc[1:].reset_index(drop=True)
        return df.apply(pd.to_numeric, errors='coerce')

    @staticmethod
    def calculate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate summary statistics for each column."""
        return df.describe()

    @staticmethod
    def detect_outliers(df: pd.DataFrame, columns: List[str]) -> dict:
        """Detect outliers using IQR method."""
        outliers = {}
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
        return outliers


class Plotter:
    """Handles all visualization functionality."""

    def __init__(self, config: DashboardConfig):
        self.config = config
        sns.set_style(config.style)
        sns.set_palette(config.palette)

    def create_time_series(self, df: pd.DataFrame, time_columns: List[str]) -> plt.Figure:
        """Generate time series plot with confidence intervals."""
        fig, ax = plt.subplots(figsize=(self.config.fig_width, self.config.fig_height))

        for col in time_columns:
            sns.lineplot(data=df, x=df.index, y=col, label=col, marker='o', ci=95)

        plt.title("Convergence Times for GSDTO Parameters", fontweight='bold')
        plt.xlabel("Index", fontweight='bold')
        plt.ylabel("Time", fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        return fig

    def create_distribution_grid(self, df: pd.DataFrame, columns: List[str]) -> plt.Figure:
        """Generate enhanced distribution analysis grid with statistical tests and detailed insights.

        Features:
        - Histogram with KDE and rugplot
        - Enhanced box plot with individual points
        - Q-Q plot with reference line
        - Statistical annotations including skewness, kurtosis, and normality tests
        """
        n_cols = len(columns)
        fig, axes = plt.subplots(3, n_cols,
                                 figsize=(self.config.fig_width * 2, self.config.fig_height * 3))

        # Ensure axes is 2D even with single column
        if n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i, col in enumerate(columns):
            data = df[col].dropna()

            # Histogram with KDE and rugplot
            sns.histplot(data=df, x=col, kde=True, ax=axes[0, i])
            sns.rugplot(data=df, x=col, ax=axes[0, i], color='red', alpha=0.5)

            # Add statistical annotations
            skewness = stats.skew(data)
            kurtosis = stats.kurtosis(data)
            _, normality_p = stats.normaltest(data)

            stats_text = (f'Skewness: {skewness:.2f}\n'
                          f'Kurtosis: {kurtosis:.2f}\n'
                          f'Normal (p): {normality_p:.3f}')

            axes[0, i].text(0.95, 0.95, stats_text,
                            transform=axes[0, i].transAxes,
                            verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            axes[0, i].set_title(f'{col} Distribution')

            # Enhanced box plot
            sns.boxplot(data=df, y=col, ax=axes[1, i], width=0.5)
            sns.swarmplot(data=df, y=col, ax=axes[1, i], color='red', size=4, alpha=0.5)

            # Add quantile annotations
            quantiles = np.percentile(data, [25, 50, 75])
            iqr = quantiles[2] - quantiles[0]
            stats_text = (f'Median: {quantiles[1]:.2f}\n'
                          f'IQR: {iqr:.2f}\n'
                          f'Range: {data.max() - data.min():.2f}')

            axes[1, i].text(0.95, 0.95, stats_text,
                            transform=axes[1, i].transAxes,
                            verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            axes[1, i].set_title(f'{col} Box Plot')

            # Q-Q plot
            (quantiles, values), (slope, intercept, r) = stats.probplot(data, dist="norm")
            axes[2, i].scatter(quantiles, values, color='blue', alpha=0.5)

            # Add reference line
            line = slope * quantiles + intercept
            axes[2, i].plot(quantiles, line, 'r', lw=2)

            # Add R² value
            r_squared = r ** 2
            axes[2, i].text(0.95, 0.05, f'R² = {r_squared:.3f}',
                            transform=axes[2, i].transAxes,
                            verticalalignment='bottom',
                            horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            axes[2, i].set_title(f'{col} Q-Q Plot')
            axes[2, i].set_xlabel('Theoretical Quantiles')
            axes[2, i].set_ylabel('Sample Quantiles')

            # Enhance all plots
            for ax in axes[:, i]:
                ax.grid(True, alpha=0.3)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        # Adjust layout
        plt.tight_layout()
        return fig

    def create_correlation_matrix(self, df: pd.DataFrame) -> plt.Figure:
        """Generate detailed correlation analysis."""
        fig, (ax1, ax2) = plt.subplots(1, 2,
                                       figsize=(self.config.fig_width * 2, self.config.fig_height))

        # Correlation heatmap
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
                    center=0, fmt='.2f', ax=ax1)
        ax1.set_title("Correlation Heatmap")

        # Correlation network plot
        mask = np.zeros_like(correlation_matrix)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(correlation_matrix, mask=mask, annot=True,
                    cmap='coolwarm', center=0, fmt='.2f', ax=ax2)
        ax2.set_title("Correlation Network")

        plt.tight_layout()
        return fig

    def create_radar_plot(self, df: pd.DataFrame, value_columns: List[str]) -> plt.Figure:
        """Generate enhanced radar plot with statistics."""
        fig, (ax1, ax2) = plt.subplots(1, 2,
                                       figsize=(self.config.fig_width * 2, self.config.fig_height),
                                       subplot_kw=dict(projection='polar'))

        # Setup angles
        angles = [n / float(len(value_columns)) * 2 * pi for n in range(len(value_columns))]
        angles += angles[:1]

        # Individual samples
        for idx, row in df.iterrows():
            values = row[value_columns].values.flatten().tolist()
            values += values[:1]
            ax1.plot(angles, values, linewidth=1, linestyle='solid', label=f'Sample {idx}')
            ax1.fill(angles, values, alpha=0.1)

        # Statistics
        mean_values = df[value_columns].mean().values.flatten().tolist()
        mean_values += mean_values[:1]
        std_values = df[value_columns].std().values.flatten().tolist()
        std_values += std_values[:1]

        ax2.plot(angles, mean_values, linewidth=2, linestyle='solid', label='Mean')
        ax2.fill_between(angles,
                         np.array(mean_values) - np.array(std_values),
                         np.array(mean_values) + np.array(std_values),
                         alpha=0.2)

        for ax in [ax1, ax2]:
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(value_columns, fontweight='bold')  

        ax1.set_title("Individual Samples", fontweight='bold')  
        ax2.set_title("Mean ± Std Dev", fontweight='bold')  
        plt.tight_layout()
        return fig

    def create_pca_visualization(self, df: pd.DataFrame, value_columns: List[str]) -> Tuple[plt.Figure, PCA]:
        """Generate a comprehensive PCA visualization with enhanced visuals for scree plot,
        scatter plot of first two components, loading heatmap, and biplot for detailed analysis."""

        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[value_columns])

        # Initialize PCA
        pca = PCA()
        pca_result = pca.fit_transform(scaled_data)

        # Set up subplots
        fig, axes = plt.subplots(2, 2, figsize=(self.config.fig_width * 2, self.config.fig_height * 2))
        fig.suptitle('PCA Analysis', fontsize=16, fontweight='bold')

        # Scree plot with explained and cumulative variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        axes[0, 0].bar(range(1, len(explained_variance) + 1), explained_variance, color='skyblue',
                       label='Explained Variance')
        axes[0, 0].plot(range(1, len(explained_variance) + 1), cumulative_variance, 'r-', marker='o',
                        label='Cumulative Variance')
        axes[0, 0].set_xlabel('Principal Components', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Variance Ratio', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Scree Plot', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, linestyle='--', alpha=0.6)

        # Scatter plot of the first two principal components
        scatter = axes[0, 1].scatter(pca_result[:, 0], pca_result[:, 1],
                                     c=df[value_columns].mean(axis=1), cmap='viridis', edgecolor='k', alpha=0.7)
        axes[0, 1].set_xlabel('PC1', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('PC2', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('First Two Principal Components', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=axes[0, 1], label='Average Feature Value' , fontweight='bold')
        cbar.ax.tick_params(labelsize=10 , fontweight='bold')
        axes[0, 1].grid(True, linestyle='--', alpha=0.6)

        # Loadings heatmap
        loadings = pca.components_.T
        loading_matrix = pd.DataFrame(loadings, columns=[f'PC{i + 1}' for i in range(loadings.shape[1])],
                                      index=value_columns)
        sns.heatmap(loading_matrix, cmap='coolwarm', center=0, annot=True, fmt='.2f', ax=axes[1, 0],
                    cbar_kws={'shrink': 0.7})
        axes[1, 0].set_title('PCA Loadings Heatmap', fontsize=14, fontweight='bold')

        # Biplot for the first two principal components
        coeff = np.transpose(pca.components_[:2, :])
        xs, ys = pca_result[:, 0], pca_result[:, 1]

        for i, (x, y) in enumerate(zip(coeff[:, 0], coeff[:, 1])):
            axes[1, 1].arrow(0, 0, x, y, color='red', alpha=0.7, linewidth=1.5, head_width=0.05)
            axes[1, 1].text(x * 1.15, y * 1.15, value_columns[i], color='darkred', ha='center', fontsize=10)

        axes[1, 1].scatter(xs, ys, alpha=0.5, color='blue', edgecolor='k')
        axes[1, 1].set_xlabel('PC1', fontsize=12, fontweight='bold')
        axes[1, 1].set_ylabel('PC2', fontsize=12, fontweight='bold')
        axes[1, 1].set_title('Biplot of Principal Components', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig, pca

    def create_cluster_plot(self, df: pd.DataFrame, value_columns: List[str]) -> plt.Figure:
        """Generate cluster analysis visualization."""
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[value_columns])
        num_samples = scaled_data.shape[0]
        from sklearn.manifold import TSNE

        # Set perplexity to a lower value if the dataset is small
        perplexity = min(30, num_samples - 1)  # Example: set to 30 or lower if dataset is smaller

        # Initialize and fit TSNE with the updated perplexity
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        tsne_results = tsne.fit_transform(scaled_data)


        fig, axes = plt.subplots(1, 2, figsize=(self.config.fig_width * 2, self.config.fig_height))

        # Dendrogram
        linkage_matrix = linkage(scaled_data, method='ward')
        dendrogram(linkage_matrix, ax=axes[0])
        axes[0].set_title('Hierarchical Clustering Dendrogram')

        # 2D embedding using t-SNE

        scatter = axes[1].scatter(tsne_results[:, 0], tsne_results[:, 1],
                                  c=df[value_columns].mean(axis=1))
        axes[1].set_title('t-SNE Visualization')
        plt.colorbar(scatter, ax=axes[1])

        plt.tight_layout()
        return fig


class GSDTODashboard:
    """Main dashboard application class."""

    def __init__(self):
        st.set_page_config(layout="wide")
        st.title("Sensitivity Analysis")
        self.config = self._setup_sidebar()
        self.plotter = Plotter(self.config)

    def _setup_sidebar(self) -> DashboardConfig:
        """Configure dashboard settings via sidebar."""
        st.sidebar.title("Visualization Settings")

        config = DashboardConfig(
            fig_height=st.sidebar.slider("Figure Height", 4, 12, 6),
            fig_width=st.sidebar.slider("Figure Width", 6, 15, 10),
            style=st.sidebar.selectbox(
                "Plot Style",
                ["darkgrid", "whitegrid", "dark", "white", "ticks"],
                index=0
            ),
            palette=st.sidebar.selectbox(
                "Color Palette",
                ["deep", "muted", "pastel", "bright", "dark", "colorblind"],
                index=0
            )
        )
        return config

    def display_summary_statistics(self, df: pd.DataFrame):
        """Display summary statistics in an expandable section."""
        with st.expander("Summary Statistics", expanded=False):
            stats_df = DataProcessor.calculate_summary_statistics(df)
            st.dataframe(stats_df)

    def display_outlier_analysis(self, df: pd.DataFrame, columns: List[str]):
        """Display outlier analysis in an expandable section."""
        with st.expander("Outlier Analysis", expanded=False):
            # Customize outlier detection or analysis approach
            outliers = DataProcessor.detect_outliers(df, columns)
            for col, indices in outliers.items():
                if indices:
                    st.write(f"Outliers detected in {col}:")
                    st.write(f"Indices: {indices}")
                    st.write(f"Values: {df.loc[indices, col].tolist()}")

    def run(self):
        """Run the dashboard application."""
        uploaded_file = st.file_uploader("Upload Excel File", type="xlsx")

        if uploaded_file is None:
            return

        try:
            # Load and process data
            df = pd.read_excel(uploaded_file, skiprows=1)
            df = DataProcessor.process_excel_data(df)

            # Get column categories
            time_cols = [col for col in df.columns if 'Time' in col]
            value_cols = [col for col in df.columns if 'Values' in col]

            # Display summary statistics and outlier analysis
            self.display_summary_statistics(df)
            self.display_outlier_analysis(df, df.columns.tolist())

            # Column selection
            selected_time_cols = st.multiselect(
                "Select Time Columns to Plot",
                time_cols,
                default=time_cols
            )

            # Create visualization tabs
            tabs = st.tabs([
                "Time Series",
                "Distributions",
                "Correlations",
                "Parameter Space",
                "PCA Analysis",
                "Cluster Analysis"
            ])

            # Populate tabs with visualizations
            with tabs[0]:
                st.subheader("Convergence Time Analysis")
                st.pyplot(self.plotter.create_time_series(df, selected_time_cols))

            with tabs[1]:
                st.subheader("Distribution Analysis")
                st.pyplot(self.plotter.create_distribution_grid(df, value_cols))

            with tabs[2]:
                st.subheader("Correlation Analysis")
                st.pyplot(self.plotter.create_correlation_matrix(df))

            with tabs[3]:

                st.subheader("Parameter Space Analysis")
                st.pyplot(self.plotter.create_radar_plot(df, value_cols))

            with tabs[4]:
                st.subheader("PCA Analysis")
                fig, pca = self.plotter.create_pca_visualization(df, value_cols)
                st.pyplot(fig)

                # Display PCA details
                with st.expander("PCA Details", expanded=False):
                    explained_variance = pca.explained_variance_ratio_
                    cumulative_variance = np.cumsum(explained_variance)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Explained Variance Ratio:")
                        for i, var in enumerate(explained_variance):
                            st.write(f"PC{i + 1}: {var:.3f}")

                    with col2:
                        st.write("Cumulative Explained Variance:")
                        for i, var in enumerate(cumulative_variance):
                            st.write(f"PC1 to PC{i + 1}: {var:.3f}")

            with tabs[5]:
                st.subheader("Cluster Analysis")
                st.pyplot(self.plotter.create_cluster_plot(df, value_cols))

                # Additional clustering insights
                with st.expander("Clustering Insights", expanded=False):
                    from scipy.cluster.hierarchy import fcluster, linkage

                    # Let user choose number of clusters
                    n_clusters = st.slider("Number of Clusters", 2, 10, 3)

                    # Perform clustering
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(df[value_cols])
                    linkage_matrix = linkage(scaled_data, method='ward')
                    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

                    # Display cluster statistics
                    cluster_df = pd.DataFrame(index=value_cols)
                    for i in range(1, n_clusters + 1):
                        cluster_data = df[clusters == i][value_cols]
                        cluster_df[f'Cluster {i} Mean'] = cluster_data.mean()
                        cluster_df[f'Cluster {i} Std'] = cluster_data.std()

                    st.write("Cluster Statistics:")
                    st.dataframe(cluster_df)

                # Add advanced analysis options
            st.sidebar.markdown("---")
            st.sidebar.subheader("Advanced Analysis Options")

            if st.sidebar.checkbox("Show Statistical Tests"):
                with st.expander("Statistical Tests", expanded=True):
                    # Normality tests
                    st.write("### Normality Tests (Shapiro-Wilk)")
                    for col in value_cols:
                        statistic, p_value = stats.shapiro(df[col].dropna())
                        st.write(f"{col}:")
                        st.write(f"- Statistic: {statistic:.3f}")
                        st.write(f"- P-value: {p_value:.3f}")
                        st.write(f"- Normal distribution: {p_value > 0.05}")

                    # Correlation tests
                    st.write("### Correlation Tests (Pearson)")
                    corr_matrix = df[value_cols].corr()
                    sig_correlations = []

                    for i in range(len(value_cols)):
                        for j in range(i + 1, len(value_cols)):
                            col1, col2 = value_cols[i], value_cols[j]
                            r, p = stats.pearsonr(df[col1].dropna(), df[col2].dropna())
                            if p < 0.05:
                                sig_correlations.append({
                                    'Variables': f"{col1} vs {col2}",
                                    'Correlation': r,
                                    'P-value': p
                                })

                    if sig_correlations:
                        st.write("Significant correlations:")
                        st.dataframe(pd.DataFrame(sig_correlations))

            if st.sidebar.checkbox("Show Time Series Analysis"):
                with st.expander("Time Series Analysis", expanded=True):
                    # Trend analysis
                    st.write("### Trend Analysis")
                    for col in selected_time_cols:
                        result = stats.linregress(df.index, df[col].dropna())
                        st.write(f"{col}:")
                        st.write(f"- Slope: {result.slope:.3f}")
                        st.write(f"- R-squared: {result.rvalue ** 2:.3f}")
                        st.write(f"- P-value: {result.pvalue:.3f}")

            # Export options
            st.sidebar.markdown("---")
            st.sidebar.subheader("Export Options")

            # Add download buttons
            csv = df.to_csv(index=False)
            st.sidebar.download_button(
                label="Download processed data (CSV)",
                data=csv,
                file_name='processed_gsdto_data.csv',
                mime='text/csv'
            )

            # Export analysis report
            if st.sidebar.button("Generate Analysis Report"):
                report = f"""
                        # GSDTO Analysis Report

                        ## Dataset Overview
                        - Number of samples: {len(df)}
                        - Number of parameters: {len(value_cols)}
                        - Time series length: {len(time_cols)}

                        ## Summary Statistics
                        {DataProcessor.calculate_summary_statistics(df).to_markdown()}

                        ## Key Findings
                        1. Time Series Analysis:
                           - Average convergence time: {df[time_cols].mean().mean():.2f}
                           - Maximum convergence time: {df[time_cols].max().max():.2f}

                        2. Parameter Analysis:
                           - Primary correlations identified
                           - Distribution characteristics noted

                        3. PCA Results:
                           - Explained variance ratios
                           - Principal component interpretations
                        """

                st.sidebar.download_button(
                    label="Download Analysis Report",
                    data=report,
                    file_name='gsdto_analysis_report.md',
                    mime='text/markdown'
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    dashboard = GSDTODashboard()
    dashboard.run()

