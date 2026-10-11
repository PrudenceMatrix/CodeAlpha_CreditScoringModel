"""
Credit scoring model my CodeAlpha internship, Task 1 on the one mmoth instanship program )

This script reads the German Credit dataset, creates a few extra
features, trains three models and compares them using precision,
recall, F1-score and ROC-AUC.

"Positive" in every metric below means a BAD credit risk, because
that is the case as i reserched a bank most wants to catch.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

DATA_FILE = Path("data") / "german.data"
RESULTS_FOLDER = Path("results")
RANDOM_SEED = 42

# The raw file has no header, so I name the columns myself.
# Names follow the description page of the dataset on UCI.
COLUMN_NAMES = [
    "checking_account_status",
    "loan_duration_months",
    "credit_history",
    "loan_purpose",
    "loan_amount",
    "savings_account_status",
    "employment_length",
    "installment_rate_percent_of_income",
    "personal_status_and_sex",
    "other_debtors_or_guarantors",
    "years_at_current_residence",
    "property_type",
    "age_in_years",
    "other_installment_plans",
    "housing_type",
    "existing_credits_at_this_bank",
    "job_type",
    "people_liable_for_maintenance",
    "has_telephone",
    "is_foreign_worker",
    "credit_risk_label",  # 1 = good, 2 = bad
]


def load_credit_data():
    credit_table = pd.read_csv(DATA_FILE, sep=" ", header=None, names=COLUMN_NAMES)

    # Turn the 1/2 label into 0/1 where 1 means "bad credit risk".
    credit_table["is_bad_credit_risk"] = (credit_table["credit_risk_label"] == 2).astype(int)
    credit_table = credit_table.drop(columns=["credit_risk_label"])
    return credit_table


def add_new_features(credit_table):
    credit_table = credit_table.copy()

    # Roughly how much the person must pay each month.
    credit_table["estimated_monthly_payment"] = (
        credit_table["loan_amount"] / credit_table["loan_duration_months"]
    )

    # A big loan is riskier for a young person than for an older one.
    credit_table["loan_amount_per_year_of_age"] = (
        credit_table["loan_amount"] / credit_table["age_in_years"]
    )

    # Flag for loans longer than two years.
    credit_table["is_long_loan"] = (credit_table["loan_duration_months"] > 24).astype(int)

    # I leave out sex and marital status on purpose. A real bank should
    # not judge credit using them, so my model should not either.
    credit_table = credit_table.drop(columns=["personal_status_and_sex"])
    return credit_table


def build_preprocessor(feature_table):
    text_columns = feature_table.select_dtypes(include="object").columns.tolist()
    number_columns = feature_table.select_dtypes(exclude="object").columns.tolist()

    return ColumnTransformer(
        [
            ("numbers", StandardScaler(), number_columns),
            ("categories", OneHotEncoder(handle_unknown="ignore"), text_columns),
        ]
    )


def build_models(feature_table):
    # Each model gets its own copy of the preprocessing steps.
    return {
        "Logistic Regression": Pipeline(
            [
                ("preprocessing", build_preprocessor(feature_table)),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "Decision Tree": Pipeline(
            [
                ("preprocessing", build_preprocessor(feature_table)),
                (
                    "model",
                    DecisionTreeClassifier(
                        max_depth=5, class_weight="balanced", random_state=RANDOM_SEED
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("preprocessing", build_preprocessor(feature_table)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300, class_weight="balanced", random_state=RANDOM_SEED
                    ),
                ),
            ]
        ),
    }


def evaluate_one_model(model_name, trained_model, test_features, test_answers):
    predicted_labels = trained_model.predict(test_features)
    predicted_bad_risk_chance = trained_model.predict_proba(test_features)[:, 1]

    print(f"\n--- {model_name} ---")
    print("Confusion matrix (rows = real, columns = predicted):")
    print(confusion_matrix(test_answers, predicted_labels))

    scores = {
        "Model": model_name,
        "Precision": precision_score(test_answers, predicted_labels),
        "Recall": recall_score(test_answers, predicted_labels),
        "F1-Score": f1_score(test_answers, predicted_labels),
        "ROC-AUC": roc_auc_score(test_answers, predicted_bad_risk_chance),
    }
    return scores, predicted_bad_risk_chance


def save_roc_curve_picture(test_answers, chances_by_model):
    plt.figure(figsize=(6, 5))
    for model_name, chances in chances_by_model.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(test_answers, chances)
        plt.plot(false_positive_rate, true_positive_rate, label=model_name)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guessing")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curves on the test data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_FOLDER / "roc_curves.png", dpi=150)
    plt.close()


def save_forest_importance_picture(random_forest_pipeline):
    feature_names = random_forest_pipeline.named_steps["preprocessing"].get_feature_names_out()
    importances = random_forest_pipeline.named_steps["model"].feature_importances_

    top_ten = (
        pd.Series(importances, index=feature_names).sort_values(ascending=False).head(10)
    )
    print("\nTop 10 features for the Random Forest:")
    print(top_ten)

    plt.figure(figsize=(8, 5))
    top_ten.sort_values().plot(kind="barh")
    plt.title("Top 10 most important features (Random Forest)")
    plt.tight_layout()
    plt.savefig(RESULTS_FOLDER / "feature_importance.png", dpi=150)
    plt.close()


def main():
    RESULTS_FOLDER.mkdir(exist_ok=True)

    credit_table = add_new_features(load_credit_data())
    print(f"Loaded {len(credit_table)} people.")
    print("Share of bad credit risks:", round(credit_table["is_bad_credit_risk"].mean(), 3))

    features = credit_table.drop(columns=["is_bad_credit_risk"])
    answers = credit_table["is_bad_credit_risk"]

    train_features, test_features, train_answers, test_answers = train_test_split(
        features, answers, test_size=0.2, stratify=answers, random_state=RANDOM_SEED
    )

    all_scores = []
    chances_by_model = {}
    trained_models = {}

    for model_name, model in build_models(features).items():
        model.fit(train_features, train_answers)
        trained_models[model_name] = model
        scores, chances = evaluate_one_model(model_name, model, test_features, test_answers)
        all_scores.append(scores)
        chances_by_model[model_name] = chances

    results_table = pd.DataFrame(all_scores).round(3)
    print("\n=== Model comparison ===")
    print(results_table.to_string(index=False))

    results_table.to_csv(RESULTS_FOLDER / "model_comparison.csv", index=False)
    save_roc_curve_picture(test_answers, chances_by_model)
    save_forest_importance_picture(trained_models["Random Forest"])
    print("\nSaved the table and pictures in the results folder.")


if __name__ == "__main__":
    main()