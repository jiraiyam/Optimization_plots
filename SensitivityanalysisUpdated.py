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

st.set_page_config(layout="wide")


def process_excel_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df.columns = [
        df.columns[i - 1] if 'Unnamed' in col else col
        for i, col in enumerate(df.columns)
    ]

    df.columns = [
        f'{col}_Values' if i % 2 == 0 else f'{col}_Time'
        for i, col in enumerate(df.columns)
    ]

    df = df.iloc[1:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, errors='coerce')

    time_cols = [col for col in df.columns if 'Time' in col]
    value_cols = [col for col in df.columns if 'Values' in col]

    return df, time_cols, value_cols


def perform_clustering(df, value_columns, time_columns, n_clusters=4):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[value_columns + time_columns])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    return kmeans.fit_predict(df_scaled)


def radar_plot(data, categories, title):
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


def radar_plot_all(df, categories, title):
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


def main():
    st.title("Sensitivity analysis")

    uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

    if uploaded_file is not None:
        raw_df = pd.read_excel(uploaded_file, skiprows=1)
        df, time_columns, value_columns = process_excel_data(raw_df)

        # Perform clustering at the start
        n_clusters = st.sidebar.slider("Number of clusters", 2, 6, 4)
        df['Cluster'] = perform_clustering(df, value_columns, time_columns, n_clusters)

        sections = st.sidebar.selectbox(
            "Choose Analysis Section",
            ["Basic Analysis", "Radar Plots", "Statistical Analysis", "Clustering Analysis",
             "Distribution Analysis", "Advanced Visualizations"]
        )

        if sections == "Basic Analysis":
            st.header("Data Overview")
            st.dataframe(df.head())

            st.subheader("Correlation Heatmap")
            fig = plt.figure(figsize=(12, 8))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
            plt.title("Heatmap of Correlations")
            st.pyplot(fig)

        elif sections == "Radar Plots":
            st.header("Radar Plot Analysis")

            categories = value_columns + time_columns  # Only use data columns, not Cluster
            fig = radar_plot(df.loc[0, categories].values, categories, 'Radar Plot for First Row Values')
            st.pyplot(fig)

            st.subheader("Radar Plots for All Rows")
            if st.button("Generate All Radar Plots"):
                radar_plot_all(df[categories], categories, 'Radar Plot for All Rows')

        elif sections == "Statistical Analysis":
            st.header("Statistical Analysis")

            st.subheader("Q-Q Plots")
            fig = plt.figure(figsize=(20, 15))
            for i, column in enumerate(value_columns + time_columns, 1):
                plt.subplot(5, 4, i)
                stats.probplot(df[column], dist="norm", plot=plt)
                plt.title(f"Q-Q Plot of {column}")
            plt.tight_layout()
            st.pyplot(fig)

        elif sections == "Clustering Analysis":
            st.header("Clustering Analysis")

            # Cluster radar plot
            st.subheader("Cluster Radar Plot")
            cluster_means = df.groupby('Cluster').mean()
            cluster_means = cluster_means[value_columns + time_columns]  # Only use data columns
            labels = cluster_means.columns
            num_vars = len(labels)
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
            for i, row in cluster_means.iterrows():
                values = row.values.tolist()
                values += [values[0]]  # Repeat first value to close the polygon
                angles_plot = angles + [angles[0]]
                ax.plot(angles_plot, values, label=f'Cluster {i}')
                ax.fill(angles_plot, values, alpha=0.25)
            
            ax.set_xticks(angles)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            plt.title("Radar Plot of Cluster Means")
            
            # Add the legend outside the main plot area
            #fig.legend(loc="center right", bbox_to_anchor=(1.1, 0.5))  # Adjust bbox_to_anchor as needed
            plt.legend(bbox_to_anchor=(1.3, 1.1))

            st.pyplot(fig)


            # 3D scatter plot
            st.subheader("3D Scatter Plot")
            if all(col in df.columns for col in ['r1_Values', 'r1_Time', 'w1_Time']):
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(df['r1_Values'], df['r1_Time'], df['w1_Time'],
                                   c=df['Cluster'], cmap='viridis', s=50)
                ax.set_xlabel('r1_Values')
                ax.set_ylabel('r1_Time')
                ax.set_zlabel('w1_Time')
                plt.colorbar(scatter)
                plt.title("3D Scatter Plot of Clusters")
                st.pyplot(fig)

            # Pair plot
            st.subheader("Pair Plot")
            if len(value_columns) > 0:
                fig = sns.pairplot(df, vars=value_columns, hue='Cluster',
                                 palette='viridis', plot_kws={'alpha': 0.6})
                plt.suptitle("Pair Plot of Value Columns by Cluster", y=1.02)
                st.pyplot(fig)

        elif sections == "Distribution Analysis":
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
            st.subheader("Box Plots by Cluster")
            if len(time_columns) > 0:
                fig = plt.figure(figsize=(14, 8))
                for i, column in enumerate(time_columns, 1):
                    plt.subplot(3, 3, i)
                    sns.boxplot(data=df, x='Cluster', y=column, palette='Set2')
                    plt.title(f"Box Plot of {column} by Cluster")
                plt.tight_layout()
                st.pyplot(fig)

            # Violin plots
            st.subheader("Violin Plots by Cluster")
            if len(value_columns) > 0:
                fig = plt.figure(figsize=(14, 8))
                for i, column in enumerate(value_columns, 1):
                    plt.subplot(3, 3, i)
                    sns.violinplot(data=df, x='Cluster', y=column, palette='muted')
                    plt.title(f"Violin Plot of {column} by Cluster")
                plt.tight_layout()
                st.pyplot(fig)


        elif sections == "Advanced Visualizations":

            st.header("Advanced Visualizations")

            # Add tabs for better organization

            tabs = st.tabs(["Time Series Analysis",

                            "Complexity Measures", "Wavelet Analysis",

                            "Network Analysis"])

            selected_column = st.sidebar.selectbox("Select column for analysis", value_columns)

            data = df[selected_column].values

            with tabs[0]:

                st.subheader("Advanced Time Series Analysis")

                # Enhanced Moving Average with Confidence Intervals

                fig = plt.figure(figsize=(14, 8))

                windows = [3, 5, 7]

                for window in windows:
                    ma = df[selected_column].rolling(window=window).mean()

                    std = df[selected_column].rolling(window=window).std()

                    plt.plot(ma, label=f'{window}-period MA')

                    plt.fill_between(df.index, ma - 2 * std, ma + 2 * std, alpha=0.2)

                plt.plot(df[selected_column], label='Original', alpha=0.5)

                plt.title(f"Moving Average Trends with Confidence Intervals")

                plt.legend()

                st.pyplot(fig)

                # Dynamic Time Warping Matrix



                st.subheader("Poincare Plot with Density")

                fig = plt.figure(figsize=(10, 10))

                x = data[:-1]

                y = data[1:]

                plt.hist2d(x, y, bins=50, cmap='viridis')

                plt.colorbar(label='Density')

                plt.xlabel('X(t)')

                plt.ylabel('X(t+1)')

                plt.title("Poincare Plot with Density Estimation")

                st.pyplot(fig)

            with tabs[2]:

                st.subheader("Advanced Complexity Measures")

                # Sample Entropy at Multiple Scales

                st.subheader("Multiscale Sample Entropy")

                scales = range(1, 11)

                entropies = []

                for scale in scales:

                    # Coarse-graining

                    coarse_grained = np.array([np.mean(data[i:i + scale])

                                               for i in range(0, len(data) - scale + 1, scale)])

                    # Simplified sample entropy calculation

                    std = np.std(coarse_grained)

                    threshold = 0.2 * std

                    count_similar = 0

                    for i in range(len(coarse_grained) - 2):

                        for j in range(i + 1, len(coarse_grained) - 2):

                            if abs(coarse_grained[i] - coarse_grained[j]) < threshold:
                                count_similar += 1

                    entropy = -np.log(count_similar / (len(coarse_grained) * (len(coarse_grained) - 1) / 2))

                    entropies.append(entropy)

                fig = plt.figure(figsize=(10, 6))

                plt.plot(scales, entropies, '-o')

                plt.xlabel('Scale Factor')

                plt.ylabel('Sample Entropy')

                plt.title('Multiscale Sample Entropy Analysis')

                st.pyplot(fig)

            with tabs[3]:

                st.subheader("Wavelet Analysis")

                # Continuous Wavelet Transform

                st.subheader("Continuous Wavelet Transform")

                scales = np.arange(1, 128)

                # Simple Morlet wavelet

                def morlet(t, scale):

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

                fig = plt.figure(figsize=(12, 8))

                plt.imshow(np.abs(cwt), aspect='auto', cmap='viridis')

                plt.colorbar(label='Magnitude')

                plt.ylabel('Scale')

                plt.xlabel('Time')

                plt.title('Continuous Wavelet Transform')

                st.pyplot(fig)

            

if __name__ == "__main__":
    main()
