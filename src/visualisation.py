import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional


class VisualEngine:
    """
    Visualization Engine generating high-DPI chart assets for report embedding.
    """

    def __init__(self, output_dir: str = "report_assets"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Apply publication style
        style_name = 'seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default'
        plt.style.use(style_name)

    def generate_all_plots(
        self,
        df: pd.DataFrame,
        driver_cols: List[str],
        target_col: str,
        target_type: str = "continuous"
    ) -> Dict[str, str]:
        """
        Generates and saves all 3 key visualization assets.
        Returns a dictionary of image paths.
        """
        if not driver_cols:
            raise ValueError("driver_cols list cannot be empty for visualization.")

        paths = {
            "importance": self._plot_importance(df, driver_cols, target_col),
            "heatmap": self._plot_heatmap(df, driver_cols, target_col),
            "trend": self._plot_trend(df, driver_cols[0], target_col, target_type)
        }
        return paths

    def _plot_importance(self, df: pd.DataFrame, driver_cols: List[str], target_col: str) -> str:
        """Horizontal bar chart showing absolute correlation strength (|r|)."""
        filepath = os.path.join(self.output_dir, "driver_importance.png")
        
        # Calculate absolute correlations with target
        corrs = [abs(df[col].corr(df[target_col])) for col in driver_cols]
        sorted_pairs = sorted(zip(corrs, driver_cols))
        sorted_scores, sorted_names = zip(*sorted_pairs)

        fig, ax = plt.subplots(figsize=(6, 2.5), dpi=300)
        bars = ax.barh(sorted_names, sorted_scores, color='#1A365D', height=0.5)

        # Label bars with values
        for bar in bars:
            width = bar.get_width()
            val_text = f"{width:.3f}" if not np.isnan(width) else "0.000"
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                val_text,
                va='center',
                fontsize=8,
                fontweight='bold',
                color='#2D3748'
            )

        ax.set_title("Primary Driver Association Strength (|r|)", pad=10, fontweight='bold', color='#1A365D', fontsize=10)
        max_val = max([s for s in sorted_scores if not np.isnan(s)], default=1.0)
        ax.set_xlim(0, max(max_val * 1.25, 0.1))
        
        # Clean styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.grid(True, linestyle='--', alpha=0.5)
        ax.yaxis.grid(False)
        
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        return filepath

    def _plot_heatmap(self, df: pd.DataFrame, driver_cols: List[str], target_col: str) -> str:
        """Annotated correlation matrix between target and selected top drivers."""
        filepath = os.path.join(self.output_dir, "correlation_heatmap.png")
        subset_cols = [target_col] + driver_cols
        corr = df[subset_cols].corr()

        fig, ax = plt.subplots(figsize=(5.5, 2.8), dpi=300)
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            ax=ax,
            linewidths=0.5,
            annot_kws={"size": 8, "weight": "bold"}
        )
        ax.set_title("Driver Correlation Matrix", pad=10, fontweight='bold', color='#1A365D', fontsize=10)
        plt.xticks(rotation=30, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        return filepath

    def _plot_trend(self, df: pd.DataFrame, top_driver: str, target_col: str, target_type: str) -> str:
        """Direct trend plot between top driver and target variable."""
        filepath = os.path.join(self.output_dir, "primary_trend.png")
        fig, ax = plt.subplots(figsize=(6.5, 2.8), dpi=300)

        if target_type == "categorical":
            # Box plot distribution per category
            sns.boxplot(
                x=target_col,
                y=top_driver,
                data=df,
                palette="Blues",
                ax=ax,
                width=0.4
            )
            ax.set_title(f"Distribution of '{top_driver}' across Target Categories", pad=10, fontweight='bold', color='#1A365D', fontsize=10)
        else:
            # Regression line for continuous target
            sns.regplot(
                x=top_driver,
                y=target_col,
                data=df,
                scatter_kws={'alpha': 0.4, 'color': '#2B6CB0', 's': 20},
                line_kws={'color': '#C53030', 'linewidth': 2},
                ax=ax
            )
            ax.set_title(f"Target Trend: '{target_col}' vs Primary Driver ('{top_driver}')", pad=10, fontweight='bold', color='#1A365D', fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        return filepath