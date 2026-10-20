from __future__ import annotations

from heart_disease.data_pipeline import train_models


def main() -> None:
    results = train_models()
    print("Training complete.")
    print(results)


if __name__ == "__main__":
    main()
