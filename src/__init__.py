"""
Iranian Telecom Churn Package
=============================
A package for predicting customer churn in the Iranian telecom dataset.
"""

from .dataset import ChurnDataSet, train_set, test_set, val_set, train_loader, test_loader, val_loader
from .module import ChurnModule
from .preprocess import transform_tr, transform_tv, minmax, standard, powerT, mm, st, pt

__all__ = [
    'ChurnDataSet',
    'train_set', 
    'test_set', 
    'val_set',
    'train_loader',
    'test_loader', 
    'val_loader',
    'ChurnModule',
    'transform_tr',
    'transform_tv',
    'minmax',
    'standard',
    'powerT',
    'mm',
    'st',
    'pt',
]

__version__ = '1.0.0'