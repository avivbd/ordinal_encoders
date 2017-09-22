import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted
import pandas as pd



class LabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, nan_classes=[], retain_nan=False, drop_unknown_classes=False):
        self.unknown_classes = []
        self.classes = []
        self.nan_classes = nan_classes
        self.retain_nan = retain_nan
        self.drop_unknown_classes = drop_unknown_classes
        
        
    def fit_transform(self, y):
        y = pd.Series(y, dtype=np.object);
        self.classes, indcs = np.unique(y, return_inverse=True)
        if self.retain_nan:
            indcs = pd.Series(indcs, dtype=np.object)
            indcs = self._ind_to_nan(indcs, y)
            self.classes = set(self.classes) - set(self.nan_classes)
            indcs = np.array(indcs, dtype=np.object)
        return indcs

    def fit(self, y_train):
        y_train = pd.Series(y_train, dtype=np.object)
        self.classes = pd.Series(y_train.unique())
        if self.retain_nan:
            self.classes = self.classes[np.invert(self.classes.isin(self.nan_classes))]
        

    def transform(self, y_test):
        check_is_fitted(self, 'classes')
        y_test = pd.Series(y_test, dtype=np.object)
        test_classes = pd.Series(y_test.unique())
        if self.retain_nan:
            test_classes = test_classes[np.invert(test_classes.isin(self.nan_classes))]
        invert_s = pd.Series(index=self.classes, 
                             data=self.classes.index.values, dtype=np.object)
        inds_s = y_test.map(invert_s)
        
        if self.retain_nan:
            inds_s = self._ind_to_nan(inds_s, y_test)
        
        diff = set(test_classes) - set(self.classes)
        if list(diff):
            warnings.warn("y contains new labels!")
            self.unknown_classes = diff
            is_unknown_cls = y_test.isin(self.unknown_classes)
            inds_s[is_unknown_cls] = len(self.classes)
            if self.drop_unknown_classes:
                inds_s = inds_s[np.invert(is_unknown_cls)]
                
        return inds_s.values

            

    def inverse_transform(self, y):

        #is y a numpy array?
        assert type(y).__module__ == np.__name__
        
        check_is_fitted(self, 'classes')
        out_of_range_values = np.setdiff1d(y, range(len(self.classes)))
        is_out_of_range = np.in1d(y, out_of_range_values)
        has_out_of_range = np.any(is_out_of_range)
        if has_out_of_range:
            warnings.warn("y contains unknown labels")
            z = np.empty(np.size(y), dtype=np.object)
            is_in_range = np.invert(is_out_of_range)
            z[is_in_range] = self.classes[y[is_in_range]]
            z[is_out_of_range] = "unknown"
            return z
        else:
            return self.classes[y]

    def _ind_to_nan(self, indcs, y):
        indcs[pd.isnull(y)] = np.nan
        for nanclass in self.nan_classes:
            indcs[y==nanclass] = nanclass
        return indcs    
                


        
        