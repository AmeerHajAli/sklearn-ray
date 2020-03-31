import numpy as np
from joblib import parallel_backend
from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC      

from ray.util.joblib import register_ray
register_ray()

param_space = {
    'C': np.logspace(-6, 6, 10),
    'gamma': np.logspace(-8, 8, 5),
    'tol': np.logspace(-4, -1, 10),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'class_weight': [None, 'balanced'],
}

model = SVC(kernel='rbf')
search = GridSearchCV(model, param_space, cv=5, verbose=1)
digits = load_digits()

with parallel_backend('ray'):
    search.fit(digits.data, digits.target)
