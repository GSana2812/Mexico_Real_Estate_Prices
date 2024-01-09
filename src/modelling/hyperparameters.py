from scipy.stats import randint
import numpy as np

RANDOM_FOREST_GS = [
    {
        'preprocessing__transformer__lat_lon__cluster_similarity__n_clusters': [5, 8, 10],
        'preprocessing__svd__n_components': [5, 8, 10],
        'random_forest__max_features': [4, 6, 8]
    },
    {
        'preprocessing__transformer__lat_lon__cluster_similarity__n_clusters': [10, 15],
        'preprocessing__svd__n_components': [6, 8, 10],
        'random_forest__max_features': [6, 8, 10]
    }
]

RANDOM_FOREST_RS = {
    'preprocessing__transformer__lat_lon__cluster_similarity__n_clusters':randint(low=3, high=50),
    'preprocessing__svd__n_components': np.arange(10, 80),
    'random_forest__max_features':randint(low=2, high=20),
    'random_forest__n_estimators':np.arange(10, 100,10),
    'random_forest__min_samples_split': np.arange(2, 20, 2),
    'random_forest__max_depth': [None, 3, 5, 10],
    'random_forest__min_samples_leaf': np.arange(1, 20, 2)
}