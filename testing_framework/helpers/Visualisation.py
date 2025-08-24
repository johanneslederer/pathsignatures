import pandas as pd
import matplotlib.pyplot as plt

def plot_grid_search(cv_results, x_param, y_param, score="mean_test_score", cmap="viridis"):
    """
    Plot a heatmap from GridSearchCV results.

    Parameters
    ----------
    cv_results : dict or DataFrame
        The cv_results_ attribute from a fitted GridSearchCV.
    x_param : str
        Name of the parameter for x-axis (e.g., 'param_estimator__alpha').
    y_param : str
        Name of the parameter for y-axis (e.g., 'param_features__depth').
    score : str, default="mean_test_score"
        Which column to plot.
    cmap : str
        Matplotlib colormap.
    """
    results = pd.DataFrame(cv_results)

    pivot = results.pivot_table(
        values=score,
        index=y_param,
        columns=x_param
    )

    # Flip sign if it's a neg metric (like neg_mean_squared_error)
    if pivot.max().max() < 0:
        pivot = -pivot

    plt.imshow(pivot, aspect="auto", cmap=cmap)
    plt.colorbar(label=score.replace("neg_", ""))
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title("Grid Search CV Scores")
    plt.tight_layout()
    plt.show()
