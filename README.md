# Credit Scoring Model

This is my Task 1 project for the CodeAlpha Machine Learning internship.
The goal is to predict whether a person is a good or a bad credit risk
using their past financial information.

## What this project does

A bank has to decide if it should give a loan to a person. If the person
does not pay back, the bank loses money. In this project I train three
machine learning models on old loan records so they can learn which
people are more likely to be a bad credit risk.

I compare these three models:

- Logistic Regression
- Decision Tree
- Random Forest

## The data

I used the German Credit dataset from the UCI Machine Learning Repository.
It has 1000 people and 20 pieces of information about each one, such as
the loan amount, how long the loan is for, the age, the savings account
status, the credit history and the type of job. Each person is marked as
a good or a bad credit risk.

About [PERCENT_BAD]% of the people in the data are bad credit risks, so
the data is not balanced. This matters when I read the results.

## What I did, step by step

1. **Loaded the data.** The file has no column names, so I added them
   myself using the dataset description.
2. **Made the target easy to read.** I changed the label so that 1 means
   a bad credit risk and 0 means a good one.
3. **Created new features** from the existing ones:
   - estimated monthly payment (loan amount divided by loan months)
   - loan amount per year of age
   - a flag for loans longer than 24 months
4. **Left out sex and marital status on purpose.** A real bank should not
   decide on credit using these, so I did not let my model use them.
5. **Prepared the data.** Numbers were scaled and text columns were turned
   into numbers (one-hot encoding). I did this inside a pipeline so the
   test data never leaks into the training.
6. **Split the data.** 80% for training and 20% for testing. I kept the
   same share of bad risks in both parts.
7. **Trained the three models.** I used balanced class weights so the
   models pay more attention to the smaller group (bad risks).
8. **Checked them on the test data** with Precision, Recall, F1-Score
   and ROC-AUC.

In all the scores below, "positive" means a bad credit risk, because that
is the case the bank most wants to catch.

## Results

--- Logistic Regression ---
Confusion matrix (rows = real, columns = predicted):
[[101  39]
 [ 11  49]]

--- Decision Tree ---
Confusion matrix (rows = real, columns = predicted):
[[91 49]
 [21 39]]

--- Random Forest ---
Confusion matrix (rows = real, columns = predicted):
[[121  19]
 [ 22  38]]

=== Model comparison ===
Model Precision Recall F1-Score ROC-AUC
Logistic Regression 0.557 0.817 0.662 0.809
Decision Tree 0.443 0.650 0.527 0.675
Random Forest 0.667 0.633 0.650 0.808

Top 10 features for the Random Forest:
numbers**estimated_monthly_payment 0.083167
numbers**loan_amount 0.077091
numbers**loan_amount_per_year_of_age 0.073422
categories**checking_account_status_A14 0.065454
numbers**loan_duration_months 0.063556
numbers**age_in_years 0.060498
categories**checking_account_status_A11 0.031425
numbers**installment_rate_percent_of_income 0.026964
numbers**years_at_current_residence 0.025959
categories**credit_history_A34 0.024370
dtype: float64

Saved the table and pictures in the results folder.

What the scores mean in simple words:

- **Precision:** when the model says "bad risk", how often it is right.
- **Recall:** out of all the real bad risks, how many the model found.
- **F1-Score:** one number that balances precision and recall.
- **ROC-AUC:** how well the model separates good from bad risks. 0.5 is
  random guessing and 1.0 is perfect.

The ROC curves and the most important features are saved as pictures in
the `results` folder.

## What I found

[WRITE 3 TO 4 SENTENCES: which model did best, which score you looked at
most and why, and what the top features were. I will help you write this
once I see your numbers.]

## Limits of this project

- The dataset is small (only 1000 people), so the scores can change if I
  split the data in a different way.
- I tested on one split only. Using cross-validation would give a more
  stable result.
- I did not tune the models much. This is a first working version.
- In real life, wrongly accepting a bad customer costs more than wrongly
  rejecting a good one. I did not use different costs here.

## How to run it

1. Download `german.data` from the UCI repository and put it in a folder
   called `data`.
2. Create a virtual environment and install the packages:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

3. Run the script:

```
python credit_scoring_model.py
```

The results table and pictures are saved in the `results` folder.

## Tools used

Python, pandas, NumPy, scikit-learn and Matplotlib.

## What I learned is that:

accuracy is not enough when one group is smaller than the
other. I had to look at precision, recall and ROC-AUC to understand how
well the models really worked. I also learned that cleaning the data and
creating new features took more effort than training the models. Using a
pipeline helped me keep the test data separate, and I chose to leave out
sex and marital status so the model stays fair.
