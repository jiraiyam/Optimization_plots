import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates
from scipy.cluster.hierarchy import linkage, dendrogram
import networkx as nx
import io
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

sns.set(style="whitegrid")
plt.rcParams.update(
    {'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold', 'axes.titlesize': 14})
def download_dataframe(df, file_format='csv'):
    """Generate download link for DataFrame"""
    if file_format == 'csv':
        return df.to_csv(index=True)
    else:
        output = io.BytesIO()
        try:
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Averages')
            return output.getvalue()
        except Exception as e:
            st.error(f"Error creating Excel file: {str(e)}")
            return None
def plot_average_metrics(averages_df):
    """Generate plots for average metrics analysis"""
    st.subheader('Download Averages Data')
    col1, col2 = st.columns(2)
    with col1:
        csv = download_dataframe(averages_df, 'csv')
        st.download_button(
            label="Download as CSV",
            data=csv,
            file_name="algorithm_averages.csv",
            mime="text/csv"
        )

    with col2:
        try:
            excel_file = download_dataframe(averages_df, 'excel')
            if excel_file is not None:
                st.download_button(
                    label="Download as Excel",
                    data=excel_file,
                    file_name="algorithm_averages.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        except Exception as e:
            st.warning("Excel download currently unavailable. Please use CSV format.")
    st.subheader('Averages DataFrame')
    st.dataframe(averages_df)
    st.subheader('Correlation Matrix of Metrics')
    correlation_matrix = averages_df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    metrics = ['Average Error', 'Average Select Size', 'Average Fitness', 'Time(S)']
    colors = ['skyblue', 'lightgreen', 'orange', 'purple']
    for metric, color in zip(metrics, colors):
        st.subheader(f'{metric} Comparison across Algorithms')
        fig, ax = plt.subplots(figsize=(10, 6))
        averages_df.loc[metric].sort_values().plot(kind='bar', color=color, ax=ax)
        ax.set_title(f'{metric} Comparison across Algorithms')
        ax.set_ylabel(metric)
        st.pyplot(fig)
    st.subheader('Distribution of Metrics Across Algorithms')
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.violinplot(data=averages_df.transpose(), inner="quart", palette="muted", ax=ax)
    ax.set_title('Distribution of Metrics Across Algorithms')
    st.pyplot(fig)
    st.subheader('Stacked Metrics Comparison')
    fig, ax = plt.subplots(figsize=(14, 8))
    averages_df.transpose().plot(kind='bar', stacked=True, colormap='tab20', ax=ax)
    ax.set_title('Stacked Metrics Comparison')
    st.pyplot(fig)
    st.subheader('Parallel Coordinates Analysis')
    normalized_df = (averages_df.transpose() - averages_df.transpose().min()) / (
                averages_df.transpose().max() - averages_df.transpose().min())
    normalized_df['Algorithm'] = normalized_df.index
    fig, ax = plt.subplots(figsize=(12, 8))
    parallel_coordinates(normalized_df, 'Algorithm', colormap='tab20')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), fontsize=12, title='Algorithm')
    st.pyplot(fig)
    st.subheader('Radar Plot Analysis')
    metrics = list(averages_df.index)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for algo in averages_df.columns:
        values = averages_df[algo].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=algo)
        ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    st.pyplot(fig)
    x = averages_df.loc['Best Fitness']
    y = averages_df.loc['Worst Fitness']
    size = averages_df.loc['Time(S)'] * 10
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(x, y, s=size, alpha=0.6, c=size, cmap='viridis', edgecolors="w", linewidth=0.5)
    plt.title('Bubble Chart: Best Fitness vs Worst Fitness (Bubble Size: Time)', fontsize=16, fontweight='bold')
    plt.xlabel('Best Fitness', fontsize=12, fontweight='bold')
    plt.ylabel('Worst Fitness', fontsize=12, fontweight='bold')
    plt.colorbar(scatter, label='Time (scaled)')
    st.pyplot(plt)
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=averages_df.transpose(), x=averages_df.columns, y=averages_df.loc['Best Fitness'])
    plt.title('Violin Plot of Best Fitness Across Algorithms', fontsize=16, fontweight='bold')
    plt.ylabel('Best Fitness', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=averages_df, palette="Set2", inner="quart", width=0.7)
    plt.title('Violin Plot of Algorithm Performance for Each Metric', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Metric Value', fontsize=12, fontweight='bold')
    plt.xlabel('Metric', fontsize=12, fontweight='bold')
    st.pyplot(plt)
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=averages_df, palette="Set2", width=0.7)
    plt.title('Box Plot of Algorithm Performance for Each Metric', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45)
    plt.ylabel('Metric Value', fontsize=12, fontweight='bold')
    plt.xlabel('Metric', fontsize=12, fontweight='bold')
    st.pyplot(plt)
    plt.figure(figsize=(14, 8))
    for algo in averages_df.columns:
        sns.kdeplot(averages_df[algo], label=algo, shade=True)
    plt.title('Density Distribution of Algorithm Performance', fontsize=16, fontweight='bold')
    plt.xlabel('Metric Value', fontsize=12, fontweight='bold')
    plt.ylabel('Density', fontsize=12, fontweight='bold')
    plt.legend(title="Algorithms")
    st.pyplot(plt)
    plt.figure(figsize=(14, 8))
    sns.heatmap(averages_df, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)
    plt.title('Algorithm Performance Across Metrics', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel('Metric', fontsize=12, fontweight='bold')
    st.pyplot(plt)
    df_long = averages_df.reset_index().melt(id_vars=['index'], var_name='Algorithm', value_name='Metric Value')
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='index', y='Metric Value', hue='Algorithm', data=df_long, split=True, inner="quart",
                   palette="muted")
    plt.title('Violin Plot of Algorithm Performance Across Metrics', fontsize=16, fontweight='bold')
    plt.ylabel('Metric Value', fontsize=12, fontweight='bold')
    plt.xlabel('Metric', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    Z = linkage(averages_df.T, method='ward', metric='euclidean')
    plt.figure(figsize=(14, 8))
    sns.clustermap(averages_df.T, row_cluster=True, col_cluster=False, figsize=(12, 8), cmap="coolwarm",
                   dendrogram_ratio=(0.1, 0.2), cbar_pos=(0, .2, .03, .4))
    plt.title('Hierarchical Clustering Heatmap of Algorithm Performance', fontsize=16, fontweight='bold')
    st.pyplot(plt)
    corr_matrix = averages_df.corr()
    G = nx.from_pandas_adjacency(corr_matrix)
    plt.figure(figsize=(14, 8))
    pos = nx.spring_layout(G, seed=42)  # Positioning of nodes
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='lightblue', font_size=12, font_weight='bold',
            edge_color='gray')
    plt.title('Network Graph of Algorithm Performance Similarity', fontsize=16, fontweight='bold')
    st.pyplot(plt)
    metrics = averages_df.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(12, 8))
    for algo in averages_df.index:
        values = averages_df.loc[algo].tolist()
        values += values[:1]
        plt.polar(angles, values, label=algo, linewidth=2)
    plt.title('Radar Plot of Algorithm Performance', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    st.pyplot(plt)
    plt.figure(figsize=(14, 8))
    sns.boxplot(data=averages_df.T, palette="Set2")
    plt.title('Box Plot of Algorithm Performance Metrics', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithms', fontsize=12, fontweight='bold')
    plt.ylabel('Metric Value', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.figure(figsize=(14, 8))
    sns.violinplot(data=averages_df.T, inner="stick", palette="muted")
    plt.title('Violin Plot of Algorithm Performance Metrics', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithms', fontsize=12, fontweight='bold')
    plt.ylabel('Metric Value', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    st.pyplot(plt)
    plt.figure(figsize=(14, 8))
    averages_df.T.plot(kind='bar', stacked=True, colormap="Set2")
    plt.title('Stacked Bar Plot of Algorithm Performance Metrics', fontsize=16, fontweight='bold')
    plt.xlabel('Metrics', fontsize=12, fontweight='bold')
    plt.ylabel('Performance Value', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend(title='Algorithms', bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)
    metrics = averages_df.columns.tolist()
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw=dict(polar=True))
    for algo in averages_df.index:
        values = averages_df.loc[algo].tolist()
        values += values[:1]  # To close the loop
        ax.bar(angles, values, width=0.3, alpha=0.6, label=algo)
    plt.title('Radial Bar Chart of Algorithm Performance Metrics', fontsize=16, fontweight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    st.pyplot(fig)
def plot_feature_selection(df):
    """Generate plots for feature selection analysis"""
    st.title("Feature Selection Score Analysis")

    def line_plot(df):
        plt.figure(figsize=(12, 6))
        for col in df.columns:
            plt.plot(df.index, df[col], marker='o', label=col)
        plt.xlabel("Observations", fontweight='bold')
        plt.ylabel("Feature Selection Score", fontweight='bold')
        plt.title("Trend of Feature Selection Scores Across Algorithms", fontweight='bold')
        plt.legend(loc="upper left", prop={'weight': 'bold'})
        plt.xticks(fontweight='bold', rotation=90)
        plt.yticks(fontweight='bold')
        st.pyplot(plt)

    # Box plot for distribution of feature selection scores
    def box_plot(df):
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df)
        plt.title("Distribution of Feature Selection Scores by Algorithm", fontweight='bold')
        plt.xlabel("Algorithm", fontweight='bold')
        plt.ylabel("Feature Selection Score", fontweight='bold')
        plt.xticks(fontweight='bold', rotation=90)
        plt.yticks(fontweight='bold')
        st.pyplot(plt)

    # Correlation heatmap of feature selection scores
    def heatmap(df):
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", square=True, annot_kws={"weight": "bold"})
        plt.title("Correlation Heatmap of Feature Selection Scores", fontweight='bold')
        plt.xticks(fontweight='bold', rotation=90)
        plt.yticks(fontweight='bold', rotation=0)
        st.pyplot(plt)

    # Violin plot for density distribution of feature selection scores
    def violin_plot(df):
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, inner="quartile")
        plt.title("Density Distribution of Feature Selection Scores by Algorithm", fontweight='bold')
        plt.xlabel("Algorithm", fontweight='bold')
        plt.ylabel("Feature Selection Score", fontweight='bold')
        plt.xticks(fontweight='bold', rotation=90)
        plt.yticks(fontweight='bold')
        st.pyplot(plt)

    violin_plot(df)
    # Display the plots in Streamlit



    # Radar plot of average feature selection scores
    def radar_plot(df):
        averages = df.mean()
        categories = df.columns
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        averages = np.concatenate((averages, [averages[0]]))
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.plot(angles, averages, linewidth=2, linestyle='solid')
        ax.fill(angles, averages, color='skyblue', alpha=0.4)
        ax.set_yticklabels([])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontweight='bold')
        plt.title("Radar Plot of Average Feature Selection Scores", fontweight='bold')
        st.pyplot(fig)

    # Scatter matrix plot for feature selection scores
    def scatter_matrix_plot(df):
        fig = pd.plotting.scatter_matrix(df, figsize=(14, 10), diagonal='kde', marker='o',
                                         hist_kwds={'color': 'skyblue', 'edgecolor': 'black'},
                                         s=60, alpha=0.7, range_padding=0.2)
        plt.suptitle("Scatter Matrix of Feature Selection Scores", fontweight='bold', y=1.02)
        st.pyplot(plt.gcf())  # Show the entire figure in Streamlit

    # KDE plot of feature selection scores by algorithm
    def kde_plot(df):
        plt.figure(figsize=(12, 6))
        for col in df.columns:
            sns.kdeplot(df[col], label=col, fill=True, linewidth=2)
        plt.xlabel("Feature Selection Score", fontweight='bold')
        plt.ylabel("Density", fontweight='bold')
        plt.title("KDE Plot of Feature Selection Scores by Algorithm", fontweight='bold')
        plt.legend(loc="upper left", prop={'weight': 'bold'})
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        st.pyplot(plt)

    kde_plot(df)



    def swarm_plot(df):
        plt.figure(figsize=(12, 6))
        sns.swarmplot(data=df, size=6, edgecolor='k', linewidth=0.7)
        plt.title("Swarm Plot of Feature Selection Scores", fontweight='bold')
        plt.xlabel("Algorithm", fontweight='bold')
        plt.ylabel("Feature Selection Score", fontweight='bold')
        plt.xticks(fontweight='bold', rotation=90)
        plt.yticks(fontweight='bold')
        st.pyplot(plt)



    def cdf_plot(df):
        plt.figure(figsize=(12, 6))
        for col in df.columns:
            sorted_data = np.sort(df[col])
            cdf = np.arange(len(sorted_data)) / float(len(sorted_data))
            plt.plot(sorted_data, cdf, label=col)
        plt.title("Cumulative Distribution Function of Feature Selection Scores", fontweight='bold')
        plt.xlabel("Feature Selection Score", fontweight='bold')
        plt.ylabel("CDF", fontweight='bold')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12, title='Algorithm')

        #plt.legend(loc="upper left", prop={'weight': 'bold'})
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        st.pyplot(plt)



    df_melt = df.melt(var_name="Algorithm", value_name="Score")

    # Ridge Plot function
    def ridge_plot(df_melt):
        g = sns.FacetGrid(df_melt, row="Algorithm", hue="Algorithm", aspect=4, height=1.5, palette="coolwarm")
        g.map(sns.kdeplot, "Score", fill=True)
        g.set_titles("{row_name}", size=12, weight='bold')
        g.set_xlabels("Feature Selection Score", weight='bold')
        g.set_ylabels("")
        g.fig.suptitle("Ridge Plot of Feature Selection Scores by Algorithm", fontweight='bold', y=1.02)
        st.pyplot(g)

    # Hexbin Plot function
    def hexbin_plot(df):
        plt.figure(figsize=(8, 6))
        plt.hexbin(df[df.columns[0]], df[df.columns[1]], gridsize=30, cmap='Blues')
        plt.colorbar(label="Counts")
        plt.xlabel(df.columns[0], fontweight='bold')
        plt.ylabel(df.columns[1], fontweight='bold')
        plt.title(f"Hexbin Plot of {df.columns[0]} vs {df.columns[1]}", fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        st.pyplot(plt)

    #hexbin_plot(df)
    # Density Heatmap function
    def density_heatmap(df):
        plt.figure(figsize=(8, 6))
        sns.kdeplot(x=df[df.columns[0]], y=df[df.columns[1]], fill=True, cmap="viridis", thresh=0.1)
        plt.xlabel(df.columns[0], fontweight='bold')
        plt.ylabel(df.columns[1], fontweight='bold')
        plt.title(f"Density Heatmap of {df.columns[0]} vs {df.columns[1]}", fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        st.pyplot(plt)

    # Pair Grid function
    def pair_grid(df):
        g = sns.PairGrid(df)
        g.map_upper(sns.scatterplot, color="purple")
        g.map_lower(sns.kdeplot, cmap="Purples")
        g.map_diag(sns.histplot, kde=True, color="purple")
        g.fig.suptitle("Pair Grid of Feature Selection Scores", fontweight='bold', y=1.02)
        st.pyplot(g.fig)



    def violin_plot(df):
        plt.figure(figsize=(12, 6))
        sns.violinplot(data=df, inner="quartile", scale="width", palette="coolwarm")
        plt.title("Violin Plot of Feature Selection Scores with Quartiles", fontweight='bold')
        plt.xlabel("Algorithm", fontweight='bold')
        plt.ylabel("Feature Selection Score", fontweight='bold')
        plt.xticks(fontweight='bold', rotation=90)
        plt.yticks(fontweight='bold')
        st.pyplot(plt)

    # Boxen Plot function
    def boxen_plot(df):
        plt.figure(figsize=(12, 6))
        sns.boxenplot(data=df, palette="coolwarm")
        plt.title("Boxen Plot of Feature Selection Scores", fontweight='bold')
        plt.xlabel("Algorithm", fontweight='bold')
        plt.ylabel("Feature Selection Score", fontweight='bold')
        plt.xticks(fontweight='bold', rotation=90)
        plt.yticks(fontweight='bold')
        st.pyplot(plt)

    # PCA Biplot function
    def pca_biplot(df):
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df)
        df_pca = pd.DataFrame(data=pca_result, columns=['PC1', 'PC2'])
        df_pca['Algorithm'] = df[df.columns[0]]  # Adjust if needed

        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="Algorithm", palette="viridis", s=100)
        for i, (x, y) in enumerate(zip(df_pca["PC1"], df_pca["PC2"])):
            plt.text(x, y, df.index[i], fontweight='bold')
        plt.title("PCA Biplot of Feature Selection Scores", fontweight='bold')
        plt.xlabel("Principal Component 1", fontweight='bold')
        plt.ylabel("Principal Component 2", fontweight='bold')
        plt.legend(loc="upper left", prop={'weight': 'bold'})
        st.pyplot(plt)

    # PCA Cumulative Variance function
    def pca_cumulative_variance(df):
        pca = PCA().fit(df)
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', color="b")
        plt.title("Cumulative Variance Explained by Principal Components", fontweight='bold')
        plt.xlabel("Number of Components", fontweight='bold')
        plt.ylabel("Cumulative Variance Explained", fontweight='bold')
        plt.xticks(fontweight='bold')
        plt.yticks(fontweight='bold')
        st.pyplot(plt)



    #st.subheader("PCA Biplot of Feature Selection Scores")
    #pca_biplot(df)

   # st.subheader("Cumulative Variance Explained by Principal Components")
   # pca_cumulative_variance(df)

    def hierarchical_dendrogram(df):
        correlation_matrix = df.corr()
        linked = linkage(correlation_matrix, 'ward')
        plt.figure(figsize=(12, 8))
        dendrogram(linked, labels=correlation_matrix.columns, leaf_rotation=90, leaf_font_size=10)
        plt.title("Hierarchical Clustering Dendrogram of Correlations", fontweight='bold')
        plt.xlabel("Features", fontweight='bold')
        plt.ylabel("Distance", fontweight='bold')
        st.pyplot(plt)

    hierarchical_dendrogram(df)
    # Function for Pairplot with Annotations
    def pairplot_with_annotations(df):
        g = sns.pairplot(df, hue=df.columns[0], palette="viridis", markers='o')
        for i, j in zip(*np.tril_indices_from(g.axes, -1)):
            g.axes[i, j].annotate(f'{df.iloc[i, j]:.2f}', (0.5, 0.5), textcoords='axes fraction', ha='center',
                                  fontsize=10, color='black', fontweight='bold')
        g.fig.suptitle("Pairplot of Features with Annotations", fontweight='bold', y=1.02)
        st.pyplot(g)

    # Subplots for Each Column
    def column_subplots(df):
        num_columns = len(df.columns)
        rows = (num_columns // 2) + (num_columns % 2)
        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(12, rows * 4))
        axes = axes.flatten()
        for i, column in enumerate(df.columns):
            axes[i].plot(df[column], label=column, color='blue')
            axes[i].set_title(f'{column}', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Index', fontsize=10, fontweight='bold')
            axes[i].set_ylabel('Value', fontsize=10, fontweight='bold')
            axes[i].legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Additional Plots for Each Column
    def multiple_plots_per_column(df):
        google_colors = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58', '#9E9E9E']  # Google colors

        num_columns = len(df.columns)
        rows = (num_columns // 2) + (num_columns % 2)

        # Set up the general style for the plots (without 'seaborn-whitegrid')
        sns.set(style="whitegrid")  # Default Seaborn style
        plt.rcParams.update({'axes.grid': True, 'grid.alpha': 0.3})  # Manually set gridlines for all plots

        # Line Plot
        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(14, rows * 6))
        axes = axes.flatten()
        for i, column in enumerate(df.columns):
            axes[i].plot(df[column], label=f'Line: {column}', color=google_colors[i % len(google_colors)],
                         linestyle='-', linewidth=2)
            axes[i].set_title(f'{column} - Line Plot', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Index', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Value', fontsize=12, fontweight='bold')
            axes[i].legend()
            axes[i].grid(True, linestyle='--', alpha=0.7)  # Add gridlines for better readability
        plt.tight_layout()
        st.pyplot(fig)

        # Histogram Plot
        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(14, rows * 6))
        axes = axes.flatten()
        for i, column in enumerate(df.columns):
            sns.histplot(df[column], kde=True, ax=axes[i], color=google_colors[i % len(google_colors)], bins=20)
            axes[i].set_title(f'{column} - Histogram', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Value', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('Frequency', fontsize=12, fontweight='bold')
            axes[i].grid(True, linestyle='--', alpha=0.7)  # Add gridlines
        plt.tight_layout()
        st.pyplot(fig)

        # Boxplot
        fig, axes = plt.subplots(nrows=rows, ncols=2, figsize=(14, rows * 6))
        axes = axes.flatten()
        for i, column in enumerate(df.columns):
            sns.boxplot(x=df[column], ax=axes[i], color=google_colors[i % len(google_colors)], linewidth=2)
            axes[i].set_title(f'{column} - Boxplot', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Value', fontsize=12, fontweight='bold')
            axes[i].grid(True, linestyle='--', alpha=0.7)  # Add gridlines for boxplot
        plt.tight_layout()
        st.pyplot(fig)



    def kde_plot(df):
        plt.figure(figsize=(12, 8))
        for column in df.columns:
            sns.kdeplot(df[column], label=column)
        plt.title('KDE Plot of Algorithm Scores', fontweight='bold')
        plt.legend()
        st.pyplot(plt)

    # Function for Ranking Heatmap
    def ranking_heatmap(df):
        rank_df = df.rank(axis=1, ascending=False)
        plt.figure(figsize=(12, 8))
        sns.heatmap(rank_df, cmap="YlGnBu", annot=True, fmt=".0f")
        plt.title('Ranking Heatmap of Algorithms', fontweight='bold')
        st.pyplot(plt)

    # Function for Distribution Plot for Each Algorithm
    def distribution_plot(df):
        plt.figure(figsize=(12, 8))
        for column in df.columns:
            sns.histplot(df[column], kde=True, label=column, bins=15, alpha=0.5)
        plt.title('Distribution Plot for Each Algorithm', fontweight='bold')
        plt.legend()
        st.pyplot(plt)

    # Function for Hierarchical Clustering Dendrogram
    def hierarchical_dendrogram(df):
        corr = df.corr()
        linked = linkage(corr, method='ward')
        plt.figure(figsize=(10, 7))
        dendrogram(linked, labels=corr.columns, orientation='top', distance_sort='descending', show_leaf_counts=True)
        plt.title('Hierarchical Clustering Dendrogram of Correlation Matrix', fontweight='bold')
        st.pyplot(plt)

    # Function for Heatmap with Hierarchical Clustering
    def heatmap_clustermap(df):
        corr = df.corr()
        sns.clustermap(corr, row_cluster=True, col_cluster=True, figsize=(10, 8), cmap='coolwarm', annot=True)
        st.pyplot(plt)

    def normalize_data(df):
        return df.div(df.max(axis=0), axis=1)

    # Boxplot with Swarmplot Overlay
    def box_swarm_plot(df , df_scaled):
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df, color="lightblue")
        sns.swarmplot(data=df_scaled, color="black", alpha=0.5)
        plt.title('Box Plot with Swarm Plot Overlay', fontweight='bold')
        plt.xticks(rotation=90)
        st.pyplot(plt)

    # Radar Chart for the First Row (Normalized)
    def radar_chart(df):
        normalized_df = normalize_data(df)
        categories = df.columns
        values = normalized_df.iloc[0].values
        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        ax.fill(angles, values, color='blue', alpha=0.25)
        ax.plot(angles, values, color='blue', linewidth=3)
        ax.set_yticklabels([])
        ax.set_xticks(angles)
        ax.set_xticklabels(categories, rotation=90)
        plt.title("Radar Chart for the First Iteration (Normalized)", fontweight='bold')
        st.pyplot(fig)

    # Boxplot with Violin Plot Overlay
    def box_violin_plot(df_melted):
        plt.figure(figsize=(12, 6))
        sns.violinplot(x='Algorithm', y='Performance', data=df_melted, color='lightblue', inner="quart", scale='width')
        sns.boxplot(x='Algorithm', y='Performance', data=df_melted, color='orange', width=0.3)
        plt.title(" Boxplot with Violin Plot", fontsize=18)
        plt.xlabel("Algorithm", fontsize=14)
        plt.ylabel("Performance", fontsize=14)
        st.pyplot(plt)

    # KDE Plot for Specific Algorithms
    def kde_plot_specific(df):
        plt.figure(figsize=(12, 6))
        sns.kdeplot(df['bSCWDTO'], shade=True, color="skyblue", label="bSCWDTO", alpha=0.7)
        sns.kdeplot(df['bDTO'], shade=True, color="orange", label="bDTO", alpha=0.7)
        sns.kdeplot(df['bSC'], shade=True, color="green", label="bSC", alpha=0.7)
        sns.kdeplot(df['bPSO'], shade=True, color="red", label="bPSO", alpha=0.7)
        plt.title("KDE Plot for Algorithm Performance", fontsize=16)
        plt.xlabel("Performance", fontsize=14)
        plt.ylabel("Density", fontsize=14)
        plt.legend()
        st.pyplot(plt)

    # Bubble Plot
    def bubble_plot(df):
        plt.figure(figsize=(12, 8))
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], s=df.iloc[:, -1] * 100, alpha=0.5, c='blue')
        plt.title('Bubble Plot', fontweight='bold')
        plt.xlabel(df.columns[0])
        plt.ylabel(df.columns[1])
        st.pyplot(plt)

    # Performance Trend Line Plot Across Iterations
    def trend_line_plot(df):
        plt.figure(figsize=(12, 6))
        for column in df.columns:
            plt.plot(df.index, df[column], label=column)
        plt.title("Performance Trend of Algorithms Across Iterations", fontsize=16, fontweight='bold')
        plt.xlabel("Iteration", fontsize=14)
        plt.ylabel("Performance", fontsize=14)
        plt.legend()
        st.pyplot(plt)

    def hexbin_plot(df):
        plt.figure(figsize=(10, 6))
        plt.hexbin(df['bSCWDTO'], df['bDTO'], gridsize=10, cmap='Blues')
        plt.colorbar(label='Density')
        plt.xlabel('bSCWDTO')
        plt.ylabel('bDTO')
        plt.title('Hexbin Plot for Algorithm Comparison', fontweight='bold')
        st.pyplot(plt)

    # Clustermap for Algorithm Performance Comparison
    def clustermap(df):
        plt.figure(figsize=(10, 6))
        sns.clustermap(df, annot=True, cmap='coolwarm', linewidths=1)
        plt.title("Clustermap of Algorithm Performance Comparison", fontsize=16)
        st.pyplot(plt)

    #clustermap(df)
    # Box Plot with KDE Overlay
    def box_kde_plot(df):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, width=0.5, color="lightblue", showfliers=False)
        sns.kdeplot(df['bSCWDTO'], color='black', fill=True, alpha=0.3)
        plt.title('Box Plot with KDE Overlay', fontweight='bold')
        st.pyplot(plt)

    # Stacked Bar Plot with Cumulative Sum
    def stacked_bar_plot(df):
        df_stacked = df.cumsum(axis=1)
        df_stacked.plot(kind="bar", stacked=True, figsize=(10, 6), cmap="viridis")
        plt.title('Stacked Bar Plot with Cumulative Sum', fontweight='bold')
        st.pyplot(plt)

    # Bar Plot with Error Bars and Annotations
    def bar_plot_with_error(df):
        means = df[['bSCWDTO', 'bPSO', 'bGWO']].mean()
        stds = df[['bSCWDTO', 'bPSO', 'bGWO']].std()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=means.index, y=means.values, yerr=stds.values, capsize=5, color='lightblue')

        # Add annotations for mean and std
        for i, (mean, std) in enumerate(zip(means, stds)):
            plt.text(i, mean + 0.02, f'Mean: {mean:.2f}', ha='center', fontsize=12, color='red')
            plt.text(i, mean - 0.02, f'STD: {std:.2f}', ha='center', fontsize=12, color='green')

        plt.title('Bar Plot with Error Bars + Statistical Annotations', fontweight='bold')
        st.pyplot(plt)

    def box_swarm_plot(df, figsize=(15, 8), palette='husl', show_outliers=True, show_stats=True):
        """
        Create enhanced box and swarm plots with statistical annotations for all numerical columns in a dataframe.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe containing numerical columns to plot
        figsize : tuple, optional
            Figure size (width, height) in inches
        palette : str, optional
            Color palette for the plots
        show_outliers : bool, optional
            Whether to highlight outliers
        show_stats : bool, optional
            Whether to show statistical annotations
        """
        # Filter numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns

        # Calculate the number of rows and columns for the subplot grid
        num_columns = len(numerical_cols)
        num_rows = int(np.ceil(num_columns / 3))  # 3 columns per row

        # Create figure
        fig = plt.figure(figsize=figsize)

        # Set the style parameters manually
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.color'] = '#b0b0b0'
        plt.rcParams['axes.grid'] = True

        # Create custom color palette
        colors = sns.color_palette(palette, n_colors=num_columns)

        # Loop through each feature
        for i, (column, color) in enumerate(zip(numerical_cols, colors)):
            ax = plt.subplot(num_rows, 3, i + 1)

            # Create Box Plot with custom style
            sns.boxplot(x=df[column], color=color, width=0.4,
                        fliersize=6, boxprops={'alpha': 0.5},
                        medianprops={'color': 'red', 'linewidth': 2})

            # Add Swarm Plot
            sns.swarmplot(x=df[column], color='black', alpha=0.5, size=4)

            if show_stats:
                # Calculate statistical measures
                mean_val = np.mean(df[column])
                std_val = np.std(df[column])
                n = len(df[column])

                stats_dict = {
                    'Mean': mean_val,
                    'Median': np.median(df[column]),
                    'Std': std_val,
                    'IQR': stats.iqr(df[column]),
                    'Skewness': stats.skew(df[column]),
                    'Kurtosis': stats.kurtosis(df[column])
                }

                # Calculate confidence interval manually
                confidence = 0.95
                degrees_of_freedom = n - 1
                t_value = stats.t.ppf((1 + confidence) / 2, degrees_of_freedom)
                margin_of_error = t_value * (std_val / np.sqrt(n))
                ci_lower = mean_val - margin_of_error
                ci_upper = mean_val + margin_of_error

                # Add statistical annotations
                stat_text = '\n'.join([f'{k}: {v:.2f}' for k, v in stats_dict.items()])
                stat_text += f'\n95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]'

                # Position the text box
                bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
                plt.annotate(stat_text,
                             xy=(1.5, 0.95),
                             xycoords='axes fraction',
                             bbox=bbox_props,
                             fontsize=8,
                             va='top')

            if show_outliers:
                # Identify outliers using z-score
                z_scores = stats.zscore(df[column])
                outliers = df[column][np.abs(z_scores) > 3]

                if not outliers.empty:
                    plt.scatter([0] * len(outliers), outliers,
                                color='red', marker='*', s=100,
                                label=f'Outliers ({len(outliers)})', zorder=5)
                    plt.legend(loc='lower right', fontsize=8)

            # Customize appearance
            plt.title(f'Distribution of {column}', pad=20, fontsize=12)
            ax.grid(True, alpha=0.3)

            # Add distribution curve
            try:
                sns.kdeplot(data=df[column], color=color, ax=ax.twinx(), alpha=0.3)
            except (ValueError, np.linalg.LinAlgError):
                # Skip KDE plot if it fails (e.g., for constant values)
                pass

        # Add main title with dataset info


        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)


        # Display in Streamlit
        st.pyplot(plt)

        # Clear the current figure to prevent memory issues
        plt.close()


        # Main title and layout adjustments

    def box_swarm_plot(df):
        plt.figure(figsize=(15, 8))

        # Define colors and styling for plots and text
        box_color = '#87CEEB'  # Light blue
        swarm_color = '#333333'  # Dark gray for contrast
        mean_color = 'crimson'
        std_color = 'darkgreen'
        font_size = 10
        title_font_size = 14

        # Calculate the number of rows and columns for the subplot grid
        num_columns = len(df.columns)
        num_rows = int(np.ceil(num_columns / 3))  # Adjust to fit all columns (3 columns per row)

        # Loop through each feature and create boxplot with swarm plot
        for i, column in enumerate(df.columns):
            plt.subplot(num_rows, 3, i + 1)  # 3 columns per row, dynamically adjusted rows

            # Create Box Plot with Swarm Plot
            sns.boxplot(x=df[column], color=box_color, width=0.5, fliersize=5, linewidth=1.5)
            sns.swarmplot(x=df[column], color=swarm_color, alpha=0.6, size=3)

            # Calculate Mean and Standard Deviation
            mean = np.mean(df[column])
            std = np.std(df[column])

            # Annotate Mean and Std
            plt.axvline(mean, color=mean_color, linestyle='--', linewidth=1.2, label=f'Mean: {mean:.2f}')
            plt.axvline(mean + std, color=std_color, linestyle=':', linewidth=1, label=f'STD: {std:.2f}')
            plt.axvline(mean - std, color=std_color, linestyle=':', linewidth=1)

            # Set title for each subplot
            plt.title(f'{column}', fontsize=title_font_size, fontweight='bold')

            # Add legend
            plt.legend(loc='upper right', fontsize=font_size - 1, frameon=False)

        # Add main title
        plt.suptitle('Box Plot with Swarm Overlay and Statistical Annotations for All Features', fontsize=16,
                     fontweight='bold', color='darkblue')

        # Adjust layout for main title
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

        # Display the plot in Streamlit
        st.pyplot(plt)

    def plot_box_swarm(df):
        num_features = len(df.columns)
        cols = 3  # Set a fixed number of columns
        rows = (num_features // cols) + (num_features % cols > 0)  # Calculate number of rows needed

        plt.figure(figsize=(cols * 5, rows * 5))  # Dynamically adjust figure size based on grid

        # Loop over all columns and create combined plots
        for i, column in enumerate(df.columns):
            plt.subplot(rows, cols, i + 1)  # Dynamically create subplots

            # Box Plot
            sns.boxplot(x=df[column], color='lightblue', width=0.4)

            # Swarm Plot
            sns.swarmplot(x=df[column], color='red', size=4, alpha=0.7)

            # Calculate Mean and Standard Deviation
            mean = np.mean(df[column])
            std_dev = np.std(df[column])

            # Annotate Mean and Std Dev
            plt.text(mean, 1.05, f'Mean: {mean:.2f}\nStd Dev: {std_dev:.2f}', ha='center', color='black', fontsize=10)

            # Set title for each subplot
            plt.title(f'{column} Box and Swarm Plot')

        plt.tight_layout()
        st.pyplot(plt)

    def plot_box_swarm_violin_stats(df):
        num_features = len(df.columns)
        cols = 3  # Set a fixed number of columns
        rows = (num_features // cols) + (num_features % cols > 0)  # Calculate number of rows required

        # Set the figure size dynamically based on the grid
        plt.figure(figsize=(cols * 5, rows * 5))

        # Loop over all columns and create combined plots
        for i, column in enumerate(df.columns):
            plt.subplot(rows, cols, i + 1)  # Dynamically create subplots

            # Create Box Plot
            sns.boxplot(x=df[column], color='lightblue', width=0.4)

            # Add Swarm Plot
            sns.swarmplot(x=df[column], color='red', size=4, alpha=0.7)

            # Create Violin Plot
            sns.violinplot(x=df[column], color='lightgreen', inner="stick", alpha=0.5)

            # Calculate Mean, Median, Std Dev, Skewness, and Kurtosis
            mean = np.mean(df[column])
            median = np.median(df[column])
            std_dev = np.std(df[column])
            skewness = stats.skew(df[column].dropna())
            kurtosis = stats.kurtosis(df[column].dropna())

            # Shapiro-Wilk Test for Normality
            shapiro_test_stat, shapiro_p_value = stats.shapiro(df[column].dropna())

            # Annotate Mean, Median, Std Dev, Skewness, Kurtosis, and p-value
            plt.text(mean, 0.85, f'Mean: {mean:.2f}\nMedian: {median:.2f}\nStd Dev: {std_dev:.2f}', ha='center',
                     color='black', fontsize=10)


            # Set title and add grid
            plt.title(f'{column} Box + Swarm + Violin + Stats', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        st.pyplot(plt)
    def scale_df(df):
        scaler = StandardScaler()
        # Assuming all columns are numerical features
        df_scaled = df.copy()
        df_scaled[df.columns] = scaler.fit_transform(df)
        return df_scaled


    def plot_violin_kde(df):
        num_columns = len(df.columns)
        rows = (num_columns // 3) + (num_columns % 3 > 0)  # Calculate the number of rows needed
        cols = 3  # Keep columns fixed at 3

        # Adjust figure size based on the number of rows
        plt.figure(figsize=(12, rows * 4))

        # Loop over each feature and create a Violin Plot with KDE
        for i, column in enumerate(df.columns):
            plt.subplot(rows, cols, i + 1)  # Adjust number of rows and columns dynamically

            # Create Violin Plot with KDE
            sns.violinplot(x=df[column], color='lightblue', inner="stick", alpha=0.7)
            sns.kdeplot(df[column], color='darkblue', linewidth=2)

            # Title and grid
            plt.title(f'{column} Violin + KDE', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        st.pyplot(plt)

    # Streamlit display


    def plot_BOX_kde(df):
        num_columns = len(df.columns)
        rows = (num_columns // 3) + (num_columns % 3 > 0)  # Calculate the number of rows needed
        cols = 3  # Keep columns fixed at 3

        plt.figure(figsize=(12, rows * 4))  # Adjust height dynamically

        # Loop over each feature and create a Violin Plot with KDE
        for i, column in enumerate(df.columns):
            plt.subplot(rows, cols, i + 1)  # Adjust number of rows and columns dynamically

            # Create Box Plot
            sns.boxplot(x=df[column], color='lightblue', width=0.4)

            # Add KDE Plot
            sns.kdeplot(df[column], color='darkblue', linewidth=2)

            # Title and grid
            plt.title(f'{column} Box + KDE', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()

        st.pyplot(plt)

        # Clear the figure to free memory
        plt.close()
    def Bar_plot(df):
        means = df.mean()
        std_devs = df.std()

        # Create a Bar Plot with error bars
        plt.figure(figsize=(10, 6))
        sns.barplot(x=df.columns, y=means, yerr=std_devs, capsize=5, color='lightblue')

        plt.title('Bar Plot with Error Bars (Mean ± Std Dev)', fontsize=16)
        plt.xticks(rotation=90)
        plt.tight_layout()
        st.pyplot(plt)

        # Clear the figure to free memory
        plt.close()












































    # Example usage:
    # box_swarm_plot(df, figsize=(15, 10), palette='husl', show_outliers=True, show_stats=True)

    # Example usage:
    # box_swarm_plot(df, figsize=(15, 10), palette='husl', show_outliers=True, show_stats=True)

    df_scaled = scale_df(df)  # Scale the DataFrame

    box_kde_plot( df_scaled)

    st.subheader("Line Plot: Trend of Feature Selection Scores Across Algorithms")
    line_plot(df)

    st.subheader("Box Plot: Distribution of Feature Selection Scores by Algorithm")
    box_plot(df)

    st.subheader("Heatmap: Correlation of Feature Selection Scores")
    heatmap(df)

    st.subheader("Violin Plot: Density Distribution of Feature Selection Scores by Algorithm")
    violin_plot(df)

    st.subheader("Hierarchical Clustering Dendrogram")
    hierarchical_dendrogram(df)

    st.subheader("Pairplot of Features with Annotations")
    pairplot_with_annotations(df)



    st.subheader("Additional Plots for Each Column")
    multiple_plots_per_column(df)

    st.subheader("Violin Plot of Feature Selection Scores with Quartiles")
    violin_plot(df)

    st.subheader("Boxen Plot of Feature Selection Scores")
    boxen_plot(df)
    st.subheader("Ridge Plot: Feature Selection Scores by Algorithm")
    ridge_plot(df_melt)

    st.subheader("Hexbin Plot: Feature Selection Scores")
    hexbin_plot(df)

    st.subheader("Density Heatmap of Feature Selection Scores")
    density_heatmap(df)

    st.subheader("Pair Grid of Feature Selection Scores")
    pair_grid(df)

    st.subheader("CDF Plot: Feature Selection Scores")
    cdf_plot(df)
    st.subheader("Swarm Plot: Feature Selection Scores by Algorithm")
    swarm_plot(df)
    st.subheader("Radar Plot of Average Feature Selection Scores")
    radar_plot(df)

    st.subheader("Scatter Matrix of Feature Selection Scores")
    scatter_matrix_plot(df)

    st.subheader("KDE Plot of Feature Selection Scores by Algorithm")
    kde_plot(df)



    st.subheader("Ranking Heatmap of Algorithms")
    ranking_heatmap(df)

    st.subheader("Distribution Plot for Each Algorithm")
    distribution_plot(df)

    st.subheader("Hierarchical Clustering Dendrogram of Correlation Matrix")
    hierarchical_dendrogram(df)

    st.subheader("Heatmap with Hierarchical Clustering")
    heatmap_clustermap(df)

    #
    df_melted = df.melt(var_name='Algorithm', value_name='Performance')

    st.subheader("Box Plot with Swarm Plot Overlay")
    box_swarm_plot(df)

    st.subheader("Radar Chart for First Iteration (Normalized)")
    radar_chart(df)

    st.subheader(" Boxplot with Violin Plot")
    box_violin_plot(df_melted)

    st.subheader("KDE Plot for Specific Algorithms")
    kde_plot_specific(df)

    st.subheader("Bubble Plot")
    bubble_plot(df)

    st.subheader("Performance Trend Across Iterations")
    trend_line_plot(df)
    ##################################
    st.subheader("Hexbin Plot for Algorithm Comparison")
    hexbin_plot(df)

    # st.subheader("Clustermap of Algorithm Performance Comparison")
    # clustermap(df)

    st.subheader("Box Plot with KDE Overlay")
    box_kde_plot(df)

    st.subheader("Stacked Bar Plot with Cumulative Sum")
    stacked_bar_plot(df)

    st.subheader("Bar Plot with Error Bars + Statistical Annotations")
    bar_plot_with_error(df)








    st.subheader("Statistical Analysis of Each Feature with Enhanced Visualization")
    box_swarm_plot(df) 
    st.subheader("Detailed Box Plot with Swarm Plot for Each Feature")
    plot_box_swarm(df) 
    
    st.subheader("Combined Analysis for Each Feature")
    plot_box_swarm_violin_stats(df)

    st.title("Violin Plots with KDE")
    df_scaled = scale_df(df)  # Scale the DataFrame

    st.subheader("Visualization for Each Feature")
    plot_violin_kde(df_scaled)
    plot_BOX_kde(df_scaled)

    Bar_plot(df)
    
             


# Main app
def main():
    st.title('Algorithm Performance Analysis')

    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

    if uploaded_file is not None:
        excel_data = pd.ExcelFile(uploaded_file)
        sheet_name = st.selectbox("Select Sheet", excel_data.sheet_names)

        # Read the selected sheet into df
        df = pd.read_excel(uploaded_file, sheet_name=sheet_name, skiprows=1)

        # Drop the "Dataset" column if it exists
        if 'Dataset' in df.columns:
            df.drop(columns=['Dataset'], inplace=True)

        # Display the selected sheet's data
        st.subheader(f"Data from Sheet: {sheet_name}")
        st.write(df)  # Display the data from the selected sheet

        # Analysis type selection
        analysis_type = st.radio(
            "Select Analysis Type",
            ["Average Metrics Analysis", "Feature Selection Analysis"]
        )

        # Clean the data by dropping any rows with missing values
        df_cleaned = df.dropna()

        # Average metrics analysis across sheets (only for Average Metrics Analysis)
        if analysis_type == "Average Metrics Analysis":
            # Prepare averages dictionary to calculate the mean of the cleaned data
            labels = []
            averages_dict = {}
            for sheet in excel_data.sheet_names:
                label_df = pd.read_excel(uploaded_file, sheet_name=sheet, skiprows=0)
                labels.append(label_df.columns[0])
                df = pd.read_excel(uploaded_file, sheet_name=sheet, skiprows=1)
                df.dropna(inplace=True)
                if 'Dataset' in df.columns:
                    df.drop(columns=['Dataset'], inplace=True)
                df = df.apply(pd.to_numeric, errors='coerce')
                df_numeric = df.dropna(axis=1, how='all')
                if not df_numeric.empty:
                    averages_dict[sheet] = df_numeric.mean()

            averages_df = pd.DataFrame(averages_dict).transpose()
            averages_df.index = labels
            averages_df.index = averages_df.index.str.strip()

            # Display the average metrics analysis
            plot_average_metrics(averages_df)

        # Feature selection analysis (if selected)
        else:
            plot_feature_selection(df_cleaned)  # Pass cleaned df for analysis

if __name__ == "__main__":
    main()