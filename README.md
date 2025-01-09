# Logistic Regression Analysis with Linear and Quadratic Terms

This project provides a Python function to analyze the relationship between a numeric variable and a binary target variable using logistic regression. It fits both linear and quadratic logistic regression models, interprets the results, and visualizes the relationship.

## Purpose

This tool is designed to help understand and visualize potentially non-linear relationships between a continuous predictor and a binary outcome. By fitting a quadratic term, the model can capture U-shaped or inverted U-shaped relationships that a simple linear model would miss.

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
