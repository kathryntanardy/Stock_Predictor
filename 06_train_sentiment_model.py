import argparse
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a sentiment classification model (no pretrained models)."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        required=True,
        help="Path to CSV file containing labeled data.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Name of the column containing the text/comments (default: 'text').",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="sentiment",
        help="Name of the column containing the sentiment label (default: 'sentiment').",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="sentiment_model.joblib",
        help="Path to save the trained model (default: 'sentiment_model.joblib').",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use as test set (default: 0.2).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (set 0 or 1 to disable CV, default: 5).",
    )
    return parser.parse_args()


def load_data(path, text_col, label_col):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)

    if text_col not in df.columns:
        raise ValueError(f"Text column '{text_col}' not found in CSV columns: {df.columns.tolist()}")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV columns: {df.columns.tolist()}")

    # Basic cleaning
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str)

    return df


def build_pipeline():
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=20000,
                ngram_range=(1, 2),
                lowercase=True,
                strip_accents="unicode",
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                n_jobs=-1,
                class_weight="balanced"
            ),
        ),
    ])
    return pipeline


def main():
    try:
        args = parse_args()
        
        print("Loading data...")
        df = load_data(args.input_csv, args.text_column, args.label_column)
        print(f"Loaded {len(df)} rows")

        X = df[args.text_column]
        y = df[args.label_column]

        if args.cv_folds and args.cv_folds > 1:
            print(f"\nCross-validation ({args.cv_folds}-fold, F1_macro)...")
            model_for_cv = build_pipeline()
            cv_scores = cross_val_score(
                model_for_cv, X, y, cv=args.cv_folds, scoring="f1_macro"
            )
            print(f"Scores per fold: {cv_scores}")
            print(f"Mean F1_macro: {cv_scores.mean():.3f}")

        print("\nSplitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )

        print("Training model...")
        model = build_pipeline()
        model.fit(X_train, y_train)

        print("Evaluating...")
        y_pred = model.predict(X_test)
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        print(f"\nSaving model to {args.output_model}...")
        joblib.dump(model, args.output_model)
        print("Done.")
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()