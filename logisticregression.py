import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.stats as stats
import math
import statsmodels.api as sm

def logistic_regression_and_interpretation(df, target_var, numeric_var):
    """
    Fits logistic regression models (linear and quadratic) for a numeric variable 
    and interprets the results, including coefficients, p-values, odds ratios, 
    turning point (for quadratic models), and relationship type, and plots the results.

    Parameters:
    - df: DataFrame containing the dataset
    - target_var: Name of the target variable (binary)
    - numeric_var: Name of the numeric variable to analyze
    """

    # Prepare the data
    X = df[numeric_var]
    y = df[target_var]

    # Linear Model
    X_linear = sm.add_constant(X)
    model_linear = sm.Logit(y, X_linear).fit(disp=0)

    # Quadratic Model
    X_squared = X ** 2
    X_combined = sm.add_constant(pd.DataFrame({numeric_var: X, f"{numeric_var}_squared": X_squared}))
    model_quadratic = sm.Logit(y, X_combined).fit(disp=0)

    # Define a function to interpret a fitted model
    def interpret_model(model, variable_name, model_type):
        coef = model.params
        pvalues = model.pvalues

        print("\nModel Interpretation:")
        print("=" * 60)
        for var in coef.index:
            if var == 'const':
                continue
            print(f"\nVariable: {var}")
            print(f"Coefficient: {coef[var]:.6f}")
            print(f"P-value: {pvalues[var]:.4f}")

            significance = "Statistically significant" if pvalues[var] < 0.05 else "Not statistically significant"
            print(significance)

            if pvalues[var] < 0.05:
                odds_ratio = np.exp(coef[var])
                print(f"Odds Ratio: {odds_ratio:.6f}")

                if var == variable_name:
                    direction = "increases" if coef[var] > 0 else "decreases"
                    print(f"As {variable_name} increases, the log-odds of target=1 {direction}.")

        if model_type == 'quadratic' and f"{variable_name}_squared" in coef.index:
            lin_coef = coef[variable_name]
            sq_coef = coef[f"{variable_name}_squared"]
            if sq_coef != 0:
                turning_point = -lin_coef / (2 * sq_coef)
                print(f"\nThe turning point (minimum/maximum) is: {turning_point:.2f}")

                if lin_coef > 0 and sq_coef < 0:
                    print(f"The relationship with {variable_name} is concave (inverted U-shaped), peaking at the turning point.")
                elif lin_coef < 0 and sq_coef > 0:
                    print(f"The relationship with {variable_name} is convex (U-shaped), reaching a minimum at the turning point.")
                else:
                  print("The quadratic relationship is not a simple U or inverted U.")
            else:
              print("The quadratic term's coefficient is zero, so there's no curvature.")

        print(f"\nPseudo R-squared: {model.prsquared:.6f}")

    # Interpret the models
    print(f"Logistic Regression for {numeric_var}:")

    print("\nLinear Model Interpretation:")
    interpret_model(model_linear, numeric_var, 'linear')

    print("\nQuadratic Model Interpretation:")
    interpret_model(model_quadratic, numeric_var, 'quadratic')

    # Plotting
    balance_range = np.linspace(df[numeric_var].min(), df[numeric_var].max(), 100)
    
    # Linear Model Plotting
    X_plot_linear = sm.add_constant(balance_range)
    predicted_probs_linear = model_linear.predict(X_plot_linear)
    
    plt.figure(figsize=(12, 6)) # two plots side by side
    plt.subplot(1, 2, 1) # first plot
    plt.plot(balance_range, predicted_probs_linear, color='blue', linewidth=2, label="Linear Model Predicted Probability")
    sns.rugplot(df[df[target_var] == 0][numeric_var], height=0.1, color="orange", alpha=0.5, label=f"{target_var} = 0")
    sns.rugplot(df[df[target_var] == 1][numeric_var], height=-0.1, color="black", alpha=0.5, label=f"{target_var} = 1")
    plt.xlabel(numeric_var)
    plt.ylabel("Probability of Target=1")
    plt.title(f"{numeric_var} vs. Probability of {target_var}=1 (Linear Model)")
    plt.grid(True)
    plt.legend()
    
    # Quadratic Model Plotting
    X_plot_quadratic = sm.add_constant(pd.DataFrame({numeric_var: balance_range, f"{numeric_var}_squared": balance_range**2}))
    predicted_probs_quadratic = model_quadratic.predict(X_plot_quadratic)
    
    plt.subplot(1, 2, 2) # second plot
    plt.plot(balance_range, predicted_probs_quadratic, color='red', linewidth=2, label="Quadratic Model Predicted Probability")
    sns.rugplot(df[df[target_var] == 0][numeric_var], height=0.1, color="orange", alpha=0.5, label=f"{target_var} = 0")
    sns.rugplot(df[df[target_var] == 1][numeric_var], height=-0.1, color="black", alpha=0.5, label=f"{target_var} = 1")
    plt.xlabel(numeric_var)
    plt.ylabel("Probability of Target=1")
    plt.title(f"{numeric_var} vs. Probability of {target_var}=1 (Quadratic Model)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
