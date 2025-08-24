from sklearn.model_selection import GridSearchCV
import numpy as np
from functools import partial

from testing_framework.helpers import ProcessGenerator, Functions, Pipeline, Visualisation

#-------- Setting the parameters
#Returns "fbm_paths, time", X will be equal to their concatenation
X_generator = partial(ProcessGenerator.fbm_d_batch, hurst=0.2, n_paths=1000, n_steps=10, n_dims=2)

#Starting from "fbm_paths, time" applies the following functions in order to compute y
y_transforms = (
    partial(Functions.fbm_to_geometric_fbm, mu=0.05),
    partial(Functions.call_on_last_index_d_dim, strike=1.2),
    partial(Functions.add_randomness, std=0.05, seed=0)
)

#Initialisation of GridSearch
param_grid = {
    "features__depth": [2, 3, 4, 5],
    "estimator__alpha": np.logspace(-3, -2,10),
}
search = GridSearchCV(Pipeline.get_pipe(), param_grid, scoring="neg_mean_squared_error", cv=5)

#-------- All computations happen in here
search.fit(*Pipeline.sample_and_get_X_y(X_generator, y_transforms))

#Showing the results
print("Best params:", search.best_params_)
print("Best score:", -search.best_score_)
Visualisation.plot_grid_search(search.cv_results_, "param_features__depth", "param_estimator__alpha")