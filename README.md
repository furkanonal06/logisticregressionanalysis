# Logistic Regression Analysis with Linear and Quadratic Terms

This is an easy, ready-to-use Python function to analyze the relationship between a numeric variable and a binary target variable using logistic regression. It fits both linear and quadratic logistic regression models, interprets the results, and visualizes the relationship.

## Purpose

This function is designed to help understand and visualize potentially non-linear relationships between a continuous predictor (numeric) and a binary outcome (categorical). By fitting a quadratic term, the model can capture U-shaped or inverted U-shaped relationships that a simple linear model would miss.

## Key Features

*   Fits both linear and quadratic logistic regression models.
*   Provides detailed interpretation of model coefficients, p-values, and odds ratios (for the linear model).
*   Calculates and interprets the turning point for quadratic models.
*   Visualizes the predicted probabilities for both models, along with rug plots to show data distribution.
*   Clear and concise output for easy interpretation.

## Installation

1.  Clone the repository:

    ```bash
    git clone [invalid URL removed]
    ```

2.  Install the required libraries:

    ```bash
    pip install pandas numpy statsmodels matplotlib seaborn
    ```

## Usage

1.  Import the function:

    ```python
    from logistic_regression_analysis import logistic_regression_and_interpretation
    ```

2.  Prepare your data: Your data should be in a pandas DataFrame with a numeric predictor variable and a binary target variable (0 or 1).

3.  Call the function:

    ```python
    import pandas as pd
    import numpy as np

    # Sample DataFrame (replace with your actual data)
    np.random.seed(0)
    df = pd.DataFrame({'Balance': np.random.randint(0, 250000, 1000), 
                       'Exited': np.random.randint(0, 2, 1000)})

    numeric_var = 'Balance'
    target_var = 'Exited'

    logistic_regression_and_interpretation(df, target_var, numeric_var)
    ```
    

    Replace `'Balance'` and `'Exited'` with the actual names of your variables.

## Example Output (Illustrative)

The function will print detailed interpretations for both models, similar to this (the actual numbers will vary based on your data):
Logistic Regression for Balance:
```python
Linear Model Interpretation:
Variable: Balance
Coefficient: 0.000005
P-value: 0.0000
Statistically significant
Odds Ratio: 1.000005
As Balance increases, the log-odds of target=1 increases.

Pseudo R-squared: 0.0161

Quadratic Model Interpretation:
Variable: Balance
Coefficient: 0.000012
P-value: 0.0000
Statistically significant

Variable: Balance_squared
Coefficient: -0.00000000005
P-value: 0.0000
Statistically significant

The turning point (minimum/maximum) is: 120000.00
The relationship with Balance is concave (inverted U-shaped), peaking at the turning point.

Pseudo R-squared: 0.0178
```
It will also generate two plots side by side: one for the linear model and one for the quadratic model, showing the predicted probabilities and data distribution.

## Statistical Concepts

*   **Logistic Regression:** A statistical method for modeling the probability of a binary outcome.
*   **Odds Ratio:** In linear logistic regression, the odds ratio represents the change in the odds of the outcome for a one-unit change in the predictor.
*   **Turning Point:** In a quadratic model, the turning point is the value of the predictor where the relationship with the outcome changes direction (minimum or maximum).
*   **Pseudo R-squared:** A measure of goodness-of-fit for logistic regression models. It is not directly comparable to the R-squared in linear regression.


## License

[MIT License](LICENSE)
