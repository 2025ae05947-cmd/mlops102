from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from heart_disease.data_pipeline import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "artifacts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = load_dataset()
    print(df.head())
    print(df.describe())
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(df["age"], bins=20, kde=True, ax=axes[0])
    axes[0].set_title("Age distribution")
    sns.countplot(x=df[TARGET_COLUMN], ax=axes[1])
    axes[1].set_title("Class balance")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "eda_summary.png", dpi=200)

    corr = df[FEATURE_COLUMNS].corr()
    heatmap = sns.heatmap(corr, cmap="coolwarm", annot=False)
    heatmap.figure.tight_layout()
    heatmap.figure.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=200)

    print(f"Saved EDA visuals to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
