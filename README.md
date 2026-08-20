# DriverFind: End-to-End Key Driver Identification & Audit Pipeline

An end-to-end Python pipeline designed to load raw tabular data, automate missing value imputation and categorical encoding, isolate top predictive drivers using transparent selection algorithms, and generate executive-ready PDF reports complete with visual charts and audit trails.

---

## Architecture Overview

```
               ┌──────────────────────────────┐
               │     Raw Data (.csv, .xlsx)   │
               └──────────────┬───────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE 1: Data Preprocessor                                │
│ • ID column dropping                                      │
│ • Target type detection (Continuous vs. Categorical)      │
│ • Categorical-to-numeric encoding (Label Encoding)        │
│ • Flexible missing value imputation (Mean, Median, Mode)  │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE 2: Feature Selection Engine                         │
│ • Modes: Filter (Correlation + Mutual Info),              │
│   Embedded (Lasso L1), or Wrapper (RFE)                   │
│ • Generates full step-by-step Audit Log                   │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE 3: Visualization Engine                             │
│ • Driver Importance (|r|) Bar Chart                       │
│ • Multi-Driver Correlation Heatmap                        │
│ • Primary Trend Scatter / Distribution Box Plot           │
└─────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────┐
│ STAGE 4: PDF Report Generator (ReportLab)                 │
│ • Page 1: Executive Summary & High-Res Visual Assets      │
│ • Page 2: Mathematical Feature Elimination Audit Log      │
└───────────────────────────────────────────────────────────┘

```

---

## Core Features

* **Multi-Format Ingestion:** Supports xls, xlsx, csv or direct Pandas DataFrames.
* **Flexible Imputation:** Choice of `mean`, `median`, `mode`, `knn`, or `None` with custom missing value marker handling (`?`, `N/A`, `null`).
* **Transparent Feature Selection:** Evaluates feature importance through three distinct methodologies while logging every pruning step.
* **Audit Trail Accountability:** Produces a standardized audit table tracking exact elimination stages, metric values, and business justifications.
* **Automated Executive PDF:** Assembles publication-quality 2-page reports using ReportLab with isolated visual assets and dynamic summary statistics.

---

## Repository Structure

```text
driverfind/
├── report_assets/           # Auto-generated PNG chart assets
├── src/
│   ├── __init__.py
│   ├── preprocessor.py      # Stage 1: DataPreprocessor class
│   ├── feature_selection.py # Stage 2: FeatureSelector & Audit Engine
│   ├── visualization.py     # Stage 3: VisualEngine chart generator
│   └── pdf_generator.py     # Stage 4: PDFReportGenerator (ReportLab)
├── main.py                  # Full execution script / integration harness
├── Key_Drivers_Executive_Report.pdf  # Final compiled PDF output
├── requirements.txt         # Project dependencies
└── README.md

```

---

## Installation & Requirements

Ensure Python 3.9+ is installed. Install all dependencies using `pip`:

```bash
pip install -r requirements.txt

```

### `requirements.txt`

```text
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
reportlab>=3.6.0
openpyxl>=3.0.0

```

---

## Quickstart Guide

Run the full pipeline using the synthetic harness in `main.py`:

```bash
python main.py

```

### Basic Programmatic Usage

```python
from src.preprocess import DataPreprocessor
from src.feature_selection import FeatureSelector
from src.visualisation import VisualEngine
from src.report_gen import PDFReportGenerator

# 1. Preprocess Raw Data
preprocessor = DataPreprocessor(
    target_col="Churn",
    id_cols=["Account_ID"],
    imputation_method="knn",
    knn_neighbors=5
)
df_processed, target_type = preprocessor.fit_transform("data/customer_churn.xlsx")

# 2. Isolate Key Drivers with Audit Log
selector = FeatureSelector(
    mode="wrapper",
    target_type=target_type,
    n_features_to_select=3
)
df_pruned, audit_df = selector.fit_select(df_processed, target_col="Churn")
isolated_drivers = [col for col in df_pruned.columns if col != "Churn"]

# 3. Generate Visual Assets
visuals = VisualEngine(output_dir="report_assets")
chart_paths = visuals.generate_all_plots(
    df=df_processed,
    driver_cols=isolated_drivers,
    target_col="Churn",
    target_type=target_type
)

# 4. Assemble Executive PDF
pdf_gen = PDFReportGenerator(output_pdf_path="Executive_Driver_Report.pdf")
pdf_gen.generate_report(
    target_col="Churn",
    target_type=target_type,
    initial_feature_count=df_processed.shape[1] - 1,
    isolated_drivers=isolated_drivers,
    audit_df=audit_df,
    chart_paths=chart_paths
)

```

---

## Pipeline Configuration Reference

### Preprocessor Parameters (`DataPreprocessor`)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `target_col` | `str` | *Required* | Name of target variable to evaluate. |
| `id_cols` | `List[str]` | `None` | Columns to exclude from modeling (e.g., ID, timestamps). |
| `custom_missing_values` | `List[Any]` | `["?", "N/A", ...]` | Strings to treat as `NaN`. |
| `imputation_method` | `str` | `"median"` | Options: `"mean"`, `"median"`, `"mode"`, `"knn"`, `None`. |
| `scale_features` | `bool` | `True` | Applies `StandardScaler` to numerical features. |

### Feature Selector Parameters (`FeatureSelector`)

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `mode` | `str` | `"wrapper"` | Selection paradigm: `"filter"`, `"embedded"`, or `"wrapper"`. |
| `target_type` | `str` | `"continuous"` | Automatically set or overriden: `"continuous"` or `"categorical"`. |
| `n_features_to_select` | `int` | `3` | Final number of driver variables to isolate. |
| `corr_threshold` | `float` | `0.85` | Pairwise correlation cutoff threshold for `"filter"` mode. |


## New to Python? Getting Started Guide

If you are a business analyst, accountant, or domain expert with zero Python experience, this guide will get you running in under 5 minutes.

### Step 1: Install Python

1. Download **Python 3.9+** from [python.org](https://www.python.org/downloads/).
2. **Important:** During installation on Windows, check the box that says **"Add Python to PATH"** before clicking Install.

### Step 2: Open Your Terminal or Command Prompt

* **Windows:** Press `Win + R`, type `cmd`, and press **Enter**.
* **Mac:** Press `Cmd + Space`, type `Terminal`, and press **Enter**.

### Step 3: Set Up the Project Folder

Navigate to the folder where you downloaded `DriverFind`:

```bash
cd path/to/driverfind

```

*(Tip: On Mac/Windows, you can type `cd ` and drag the folder directly into the terminal window).*

### Step 4: Install Dependencies & Run

Copy and paste these commands into your terminal one by one:

```bash
# 1. Create a safe isolated environment (optional but recommended)
python -m venv venv

# 2. Activate the environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install required packages
pip install -r requirements.txt

# 4. Run the full pipeline!
python main.py

```

### Step 5: View Your Generated PDF

Once the script finishes running, check your `driverfind` folder. You will find a brand-new file named **`Key_Drivers_Executive_Report.pdf`** ready to open in any PDF viewer.

### How to Analyze Your Own Data File

To analyze your own Excel (`.xlsx`) or CSV file instead of synthetic data:

1. Drop your file into the project folder (e.g., `my_sales_data.xlsx`).
2. Open `main.py` in any basic text editor (like Notepad or TextEdit).
3. Change the preprocessing step to point to your file name and target variable:

```python
# Change "my_sales_data.xlsx" and "Revenue" to match your actual spreadsheet
df_processed, target_type = preprocessor.fit_transform("my_sales_data.xlsx")

```

4. Run `python main.py` again!