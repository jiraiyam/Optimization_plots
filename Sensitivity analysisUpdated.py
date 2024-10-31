import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from typing import Tuple, List
from pandas.plotting import andrews_curves, parallel_coordinates
from mpl_toolkits.mplot3d import Axes3D

# Configuration
st.set_page_config(layout="wide")

# Data Processing Functions
def process_excel_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Process the input Excel data and return processed dataframe with column lists."""
    # Fix column names
    df.columns = [
        df.columns[i - 1] if 'Unnamed' in col else col
        for i, col in enumerate(df.columns)
    ]
    
    # Add suffixes for values and time columns
    df.columns = [
        f'{col}_Values' if i % 2 == 0 else f'{col}_Time'
        for i, col in enumerate(df.columns)
    ]
    
    # Process data
    df = df.iloc[1:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Split columns
    time_cols = [col for col in df.columns if 'Time' in col]
    value_cols = [col for col in df.columns if 'Values' in col]
    
    return df, time_cols, value_cols

def perform_clustering(df: pd.DataFrame, value_columns: List[str], 
                      time_columns: List[str], n_clusters: int = 4) -> np.ndarray:
    """Perform KMeans clustering on the data."""
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[value_columns + time_columns])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(df_scaled)

# Visualization Functions
def create_radar_plot(data: np.ndarray, categories: List[str], title: str) -> plt.Figure:
    """Create a single radar plot."""
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    data = np.concatenate((data, [data[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, data, color='blue', alpha=0.25)
    ax.plot(angles, data, color='blue', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    plt.title(title, size=15, color='black', weight='bold')
    return fig

def create_radar_plots_all(df: pd.DataFrame, categories: List[str], title: str) -> None:
    """Create radar plots for all rows in the dataframe."""
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    for index, row in df.iterrows():
        row_values = row[categories].values.flatten().tolist() + [row[categories].values[0]]
        fig = plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, polar=True)
        ax.fill(angles, row_values, alpha=0.25)
        ax.plot(angles, row_values, linewidth=2)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_title(f"{title} - Row {index + 1}", size=15, color='black', weight='bold')
        st.pyplot(fig)

def create_cluster_radar_plot(df: pd.DataFrame, value_columns: List[str], 
                            time_columns: List[str]) -> plt.Figure:
    """Create radar plot for cluster means."""
    cluster_means = df.groupby('Cluster').mean()
    cluster_means = cluster_means[value_columns + time_columns]
    labels = cluster_means.columns
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, row in cluster_means.iterrows():
        values = row.values.tolist()
        values += [values[0]]
        angles_plot = angles + [angles[0]]
        ax.plot(angles_plot, values, label=f'Cluster {i}')
        ax.fill(angles_plot, values, alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    plt.title("Radar Plot of Cluster Means")
    plt.legend()
    return fig

def create_3d_scatter(df: pd.DataFrame) -> plt.Figure:
    """Create 3D scatter plot of clusters."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df['r1_Values'], df['r1_Time'], df['w1_Time'],
                        c=df['Cluster'], cmap='viridis', s=50)
    ax.set_xlabel('r1_Values')
    ax.set_ylabel('r1_Time')
    ax.set_zlabel('w1_Time')
    plt.colorbar(scatter)
    plt.title("3D Scatter Plot of Clusters")
    return fig

def calculate_wavelet_transform(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate continuous wavelet transform."""
    scales = np.arange(1, 128)
    
    def morlet(t: np.ndarray, scale: float) -> np.ndarray:
        """Morlet wavelet function."""
        return np.exp(1j * 5.0 * t / scale) * np.exp(-(t ** 2) / (2 * scale ** 2))
    
    cwt = np.zeros((len(scales), len(data)), dtype=complex)
    for i, scale in enumerate(scales):
        wavelet_data = np.zeros(len(data), dtype=complex)
        for t in range(len(data)):
            t_min = max(0, t - int(4 * scale))
            t_max = min(len(data), t + int(4 * scale))
            wavelet = morlet(np.arange(t_min - t, t_max - t), scale)
            wavelet_data[t] = np.sum(data[t_min:t_max] * wavelet)
        cwt[i, :] = wavelet_data
    return cwt, scales

def create_visibility_graph(data: np.ndarray) -> np.ndarray:
    """Create horizontal visibility graph from time series data."""
    n = len(data)
    adj_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i + 1, n):
            visible = True
            for k in range(i + 1, j):
                if data[k] >= min(data[i], data[j]):
                    visible = False
                    break
            if visible:
                adj_matrix[i, j] = adj_matrix[j, i] = 1
                
    return adj_matrix

# Analysis Section Functions
def show_basic_analysis(df: pd.DataFrame) -> None:
    """Display basic analysis section."""
    st.header("Data Overview")
    st.dataframe(df.head())
    
    st.subheader("Correlation Heatmap")
    fig = plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Heatmap of Correlations")
    st.pyplot(fig)

def show_radar_analysis(df: pd.DataFrame, value_columns: List[str], 
                       time_columns: List[str]) -> None:
    """Display radar plot analysis section."""
    st.header("Radar Plot Analysis")
    categories = value_columns + time_columns
    fig = create_radar_plot(df.loc[0, categories].values, categories, 
                          'Radar Plot for First Row Values')
    st.pyplot(fig)
    
    st.subheader("Radar Plots for All Rows")
    if st.button("Generate All Radar Plots"):
        create_radar_plots_all(df[categories], categories, 'Radar Plot for All Rows')

def show_statistical_analysis(df: pd.DataFrame, value_columns: List[str], 
                            time_columns: List[str]) -> None:
    """Display statistical analysis section."""
    st.header("Statistical Analysis")
    st.subheader("Q-Q Plots")
    fig = plt.figure(figsize=(20, 15))
    for i, column in enumerate(value_columns + time_columns, 1):
        plt.subplot(5, 4, i)
        stats.probplot(df[column], dist="norm", plot=plt)
        plt.title(f"Q-Q Plot of {column}")
    plt.tight_layout()
    st.pyplot(fig)

def show_clustering_analysis(df: pd.DataFrame, value_columns: List[str], 
                           time_columns: List[str]) -> None:
    """Display clustering analysis section."""
    st.header("Clustering Analysis")
    
    # Cluster radar plot
    st.subheader("Cluster Radar Plot")
    fig = create_cluster_radar_plot(df, value_columns, time_columns)
    st.pyplot(fig)
    
    # 3D scatter plot
    if all(col in df.columns for col in ['r1_Values', 'r1_Time', 'w1_Time']):
        st.subheader("3D Scatter Plot")
        fig = create_3d_scatter(df)
        st.pyplot(fig)
    
    # Pair plot
    st.subheader("Pair Plot")
    if len(value_columns) > 0:
        fig = sns.pairplot(df, vars=value_columns, hue='Cluster',
                          palette='viridis', plot_kws={'alpha': 0.6})
        plt.suptitle("Pair Plot of Value Columns by Cluster", y=1.02)
        st.pyplot(fig)

def show_distribution_analysis(df: pd.DataFrame, value_columns: List[str], 
                             time_columns: List[str]) -> None:
    """Display distribution analysis section."""
    st.header("Distribution Analysis")
    
    # Distribution plots
    st.subheader("Distribution Plots")
    fig = plt.figure(figsize=(20, 15))
    for i, column in enumerate(value_columns + time_columns, 1):
        plt.subplot(5, 4, i)
        sns.histplot(df[column], kde=True)
        plt.title(f"Distribution of {column}")
    plt.tight_layout()
    st.pyplot(fig)
    
    # Box plots
    if len(time_columns) > 0:
        st.subheader("Box Plots by Cluster")
        fig = plt.figure(figsize=(14, 8))
        for i, column in enumerate(time_columns, 1):
            plt.subplot(3, 3, i)
            sns.boxplot(data=df, x='Cluster', y=column, palette='Set2')
            plt.title(f"Box Plot of {column} by Cluster")
        plt.tight_layout()
        st.pyplot(fig)
    
    # Violin plots
    if len(value_columns) > 0:
        st.subheader("Violin Plots by Cluster")
        fig = plt.figure(figsize=(14, 8))
        for i, column in enumerate(value_columns, 1):
            plt.subplot(3, 3, i)
            sns.violinplot(data=df, x='Cluster', y=column, palette='muted')
            plt.title(f"Violin Plot of {column} by Cluster")
        plt.tight_layout()
        st.pyplot(fig)

def show_advanced_analysis(df: pd.DataFrame, selected_column: str) -> None:
    """Display advanced analysis section."""
    st.header("Advanced Analysis")
    data = df[selected_column].values
    
    tabs = st.tabs(["Time Series Analysis", "Wavelet Analysis", "Network Analysis"])
    
    with tabs[0]:
        st.subheader("Advanced Time Series Analysis")
        
        # Moving Average with Confidence Intervals
        fig = plt.figure(figsize=(14, 8))
        windows = [3, 5, 7]
        for window in windows:
            ma = pd.Series(data).rolling(window=window).mean()
            std = pd.Series(data).rolling(window=window).std()
            plt.plot(ma, label=f'{window}-period MA')
            plt.fill_between(range(len(data)), ma - 2 * std, ma + 2 * std, alpha=0.2)
        plt.plot(data, label='Original', alpha=0.5)
        plt.title("Moving Average Trends with Confidence Intervals")
        plt.legend()
        st.pyplot(fig)
        
        # Poincare Plot
        st.subheader("Poincare Plot with Density")
        fig = plt.figure(figsize=(10, 10))
        plt.hist2d(data[:-1], data[1:], bins=50, cmap='viridis')
        plt.colorbar(label='Density')
        plt.xlabel('X(t)')
        plt.ylabel('X(t+1)')
        plt.title("Poincare Plot with Density Estimation")
        st.pyplot(fig)
    
    with tabs[1]:
        st.subheader("Wavelet Analysis")
        cwt, scales = calculate_wavelet_transform(data)
        fig = plt.figure(figsize=(12, 8))
        plt.imshow(np.abs(cwt), aspect='auto', cmap='viridis')
        plt.colorbar(label='Magnitude')
        plt.ylabel('Scale')
        plt.xlabel('Time')
        plt.title('Continuous Wavelet Transform')
        st.pyplot(fig)
    
    with tabs[2]:
        st.subheader("Network Analysis")
        adj_matrix = create_visibility_graph(data)
        degrees = np.sum(adj_matrix, axis=0)
        fig = plt.figure(figsize=(12, 6))
        plt.hist(degrees, bins=30, density=True, alpha=0.7)
        plt.xlabel('Degree')
        plt.ylabel('Probability')
        plt.title('Degree Distribution of Visibility Graph')
        st.pyplot(fig)

def main():
    """Main application function."""
    st.title("Sensitivity Analysis")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])
    
    if uploaded_file is not None:
        # Process data
        raw_df = pd.read_excel(uploaded_file, skiprows=1)
        df, time_columns, value_columns = process_excel_data(raw_df)
        
        # Clustering setup
        n_clusters = st.sidebar.slider("Number of clusters", 2, 6, 4)
        df['Cluster'] = perform_clustering(df, value_columns, time_columns, n_clusters)
        
        # Section selection
        sections = st.sidebar.selectbox(
            "Choose Analysis Section",
            ["Basic Analysis", "Radar Plots", "Statistical Analysis"]
        )
        
        # Display selected section
        if sections == "Basic Analysis":
            show_basic_analysis(df)
        elif sections == "Radar Plots":
            show_radar_analysis(df, value_columns, time_columns)
        elif sections == "Statistical Analysis":
            show_statistical_analysis(df, value_columns, time_columns)

if __name__ == "__main__":
    main()
