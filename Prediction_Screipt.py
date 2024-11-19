import streamlit as st
import pandas as pd
from typing import Dict, List, Union
from dataclasses import dataclass


@dataclass
class ConfusionMatrix:
    """Class to represent a confusion matrix with its components."""
    a: int  # True Positives
    b: int  # False Positives
    c: int  # False Negatives
    d: int  # False Negatives


class MetricsCalculator:
    """Class to handle all metric calculations and matrix updates."""

    @staticmethod
    def calculate_metrics(matrix: ConfusionMatrix) -> Dict[str, float]:
        """Calculate evaluation metrics from a confusion matrix."""
        total = matrix.a + matrix.b + matrix.c + matrix.d

        # Calculate core metrics with safe division
        accuracy = (matrix.a + matrix.d) / total if total > 0 else 0
        sensitivity = matrix.a / (matrix.a + matrix.c) if (matrix.a + matrix.c) > 0 else 0
        specificity = matrix.d / (matrix.b + matrix.d) if (matrix.b + matrix.d) > 0 else 0
        ppv = matrix.a / (matrix.a + matrix.b) if (matrix.a + matrix.b) > 0 else 0
        npv = matrix.d / (matrix.c + matrix.d) if (matrix.c + matrix.d) > 0 else 0

        # Calculate F-score
        fscore = (2 * ppv * sensitivity) / (ppv + sensitivity) if (ppv + sensitivity) > 0 else 0

        return {
            'Model': '',  # Will be filled later
            'Accuracy': accuracy,
            'Sensitivity (TPR)': sensitivity,
            'Specificity (TNR)': specificity,
            'PPV': ppv,
            'NPV': npv,
            'F-Score': fscore
        }

    @staticmethod
    def update_matrix(matrix: ConfusionMatrix, multiplier: float) -> ConfusionMatrix:
        """Update confusion matrix values based on a multiplier."""
        return ConfusionMatrix(
            a=int(matrix.a * multiplier),
            b=max(1, int(matrix.b * (1 / multiplier))),
            c=max(1, int(matrix.c * (1 / multiplier))),
            d=int(matrix.d * multiplier)
        )


class MetricsApp:
    """Main Streamlit application class."""

    def __init__(self):
        self.stage_multipliers = {
            "Initial": 1.0,
            "Feature Selection": 1.5,
            "Optimization": 2.25
        }
        self.default_models = """Neural Network (MLP)
Random Forest
Support Vector Machine
K-Nearest Neighbors
Naive Bayes
Decision Tree
Gradient Boosting"""

    def setup_page(self):
        """Configure the Streamlit page settings."""
        st.set_page_config(
            page_title="ML Model Metrics Calculator",
            page_icon="📊",
            layout="wide"
        )

        st.title("🎯 Machine Learning Model Evaluation Calculator")
        st.markdown("""
        ### Calculate and compare evaluation metrics across different models and stages
        This tool helps you analyze model performance through various stages of development:
        - **Initial Stage**: Baseline model performance
        - **Feature Selection**: After feature engineering
        - **Optimization**: Final tuned performance
        """)

    def get_model_inputs(self) -> List[str]:
        """Collect and validate model names from user input."""
        with st.expander("📝 Enter Model Names", expanded=True):
            models = st.text_area(
                "Enter model names (one per line):",
                value=self.default_models,
                height=150,
                key="model_names_input"
            ).splitlines()
            return [model.strip() for model in models if model.strip()]

    def collect_matrix_inputs(self, models: List[str]) -> Dict[str, ConfusionMatrix]:
        """Collect confusion matrix inputs for each model."""
        matrices = {}

        st.header("📊 Confusion Matrix Inputs")
        cols = st.columns(2)

        with cols[0]:
            st.markdown("""
            ### Matrix Components:
            - **A**: True Positives (TP)
            - **B**: False Positives (FP)
            - **C**: False Negatives (FN)
            - **D**: True Negatives (TN)
            """)

        for idx, model in enumerate(models):
            with st.expander(f"🔍 {model} Confusion Matrix", expanded=True):
                col1, col2, col3, col4 = st.columns(4)

                matrices[model] = ConfusionMatrix(
                    a=col1.number_input(f"TP (A) - {model}", min_value=0, value=100, step=1, key=f"tp_{idx}"),
                    b=col2.number_input(f"FP (B) - {model}", min_value=0, value=10, step=1, key=f"fp_{idx}"),
                    c=col3.number_input(f"FN (C) - {model}", min_value=0, value=5, step=1, key=f"fn_{idx}"),
                    d=col4.number_input(f"TN (D) - {model}", min_value=0, value=85, step=1, key=f"tn_{idx}")
                )

        return matrices

    def display_results(self, results_df: pd.DataFrame, stage: str):
        """Display results and provide download option."""
        st.subheader(f"📈 {stage} Stage Results")

        # Ensure Model is the first column and sort by Accuracy
        cols = ['Model'] + [col for col in results_df.columns if col != 'Model']
        results_df = results_df[cols].sort_values('Accuracy', ascending=False)

        # Style the dataframe
        styled_df = (results_df.style
                     .format({col: "{:.4f}" for col in results_df.columns if col != "Model"})
                     .background_gradient(subset=[col for col in results_df.columns if col != "Model"],
                                          cmap='Blues'))

        # Display dataframe
        st.dataframe(styled_df, hide_index=True)

        # Download button with unique key
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"⬇️ Download {stage} Results (CSV)",
            data=csv,
            file_name=f"{stage.lower()}_metrics.csv",
            mime="text/csv",
            key=f"download_{stage.lower()}"
        )

    def run(self):
        """Main application logic."""
        self.setup_page()

        models = self.get_model_inputs()
        matrices = self.collect_matrix_inputs(models)

        # Process and display results for each stage
        calculator = MetricsCalculator()

        for stage, multiplier in self.stage_multipliers.items():
            results = []
            for model, matrix in matrices.items():
                metrics = calculator.calculate_metrics(
                    calculator.update_matrix(matrix, multiplier) if stage != "Initial" else matrix
                )
                metrics['Model'] = model  # Set model name in metrics
                results.append(metrics)

            df = pd.DataFrame(results)
            self.display_results(df, stage)

        # Optimization stage
        with st.expander("🎯 Model Optimization", expanded=True):
            selected_model = st.selectbox(
                "Select model for optimization:",
                models,
                key="optimizer_model_select"
            )
            optimizers = st.text_input(
                "Enter optimizer names (comma-separated):",
                "GA, PSO, GWO",
                key="optimizer_names"
            ).split(',')

            optimization_results = []
            for optimizer in optimizers:
                optimizer = optimizer.strip()
                matrix = matrices[selected_model]
                metrics = calculator.calculate_metrics(
                    calculator.update_matrix(matrix, self.stage_multipliers["Optimization"])
                )
                metrics['Model'] = f"{optimizer} {selected_model}"
                optimization_results.append(metrics)

            df_optimization = pd.DataFrame(optimization_results)
            self.display_results(df_optimization, "OptimizationResults")


if __name__ == "__main__":
    app = MetricsApp()
    app.run()
