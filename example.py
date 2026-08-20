import numpy as np
import pandas as pd

# Import all 4 modular stages
from src.preprocess import DataPreprocessor
from src.feature_selection import FeatureSelector
from src.visualisation import VisualEngine
from src.report_gen import PDFReportGenerator


def main():
    print("STAGE 1: LOADING & PREPROCESSING DATA",flush=True)
    # Construct synthetic dataset (or load csv/xlsx/txt etc)
    np.random.seed(42)
    n = 200
    df_raw = pd.DataFrame({
        "Account_ID": [f"ACC_{i:03d}" for i in range(n)],
        "Region": np.random.choice(["North", "South", "East", "West"], n),
        "Monthly_Cost": np.random.normal(1200, 300, n),
        "Support_Tickets": np.random.poisson(3, n),
        "Usage_Hours": np.random.normal(150, 40, n),
        "Noise_Metric_1": np.random.normal(0, 1, n),
        "Churn": np.random.choice([0, 1], n, p=[0.7, 0.3])
    })

    preprocessor = DataPreprocessor(
        target_col="Churn",
        id_cols=["Account_ID"],
        imputation_method="knn",
        knn_neighbors=3
    )
    df_processed, target_type = preprocessor.fit_transform(df_raw)

    print("\nSTAGE 2: RUNNING FEATURE SELECTION ENGINE",flush=True)
    selector = FeatureSelector(
        mode="wrapper",
        target_type=target_type,
        n_features_to_select=3
    )
    df_pruned, audit_df = selector.fit_select(df_processed, target_col="Churn")
    isolated_drivers = [c for c in df_pruned.columns if c != "Churn"]

    print("\nSTAGE 3: GENERATING CHART ASSETS",flush=True)
    visuals = VisualEngine(output_dir="report_assets")
    chart_paths = visuals.generate_all_plots(
        df=df_processed,
        driver_cols=isolated_drivers,
        target_col="Churn",
        target_type=target_type
    )

    print("\nSTAGE 4: ASSEMBLING PDF REPORT",flush=True)
    pdf_gen = PDFReportGenerator(output_pdf_path="Key_Drivers_Executive_Report.pdf")
    pdf_gen.generate_report(
        target_col="Churn",
        target_type=target_type,
        initial_feature_count=df_raw.shape[1] - 2, # Exclude target and ID
        isolated_drivers=isolated_drivers,
        audit_df=audit_df,
        chart_paths=chart_paths
    )

    print("\nPipeline execution finished successfully!",flush=True)


if __name__ == "__main__":
    main()