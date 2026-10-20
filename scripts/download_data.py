from __future__ import annotations

from heart_disease.data_pipeline import download_or_prepare_data


def main() -> None:
    path = download_or_prepare_data()
    print(f"Dataset prepared at: {path}")


if __name__ == "__main__":
    main()
