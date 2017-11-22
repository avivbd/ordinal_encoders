import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import pdb



class LabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, nan_classes=[], retain_nan=False, drop_unknown_classes=False, use_mode=False):
        self.unknown_classes = []
        self.train_classes = []
        self.nan_classes = nan_classes
        self.retain_nan = retain_nan
        self.drop_unknown_classes = drop_unknown_classes
        self.use_mode =  use_mode
        self.test_classes = []
        
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def fit(self, y_train):
        self.y_train = pd.Series(y_train, dtype=np.object)
        self.train_val_cnts = self.y_train.value_counts(dropna=False)
        self.train_classes = pd.Series(self.train_val_cnts.index.values)
        if self.retain_nan:
            train_in_nan = self.train_classes.isin(self.nan_classes)
            self.train_classes = self.train_classes[~train_in_nan]
        if self.use_mode:
            self.most_common_class = self.train_classes.iloc[:1]

    def transform(self, y_test):
        check_is_fitted(self, 'train_classes')
        self.y_test = pd.Series(y_test, dtype=np.object)
        self.test_value_cnts = self.y_test.value_counts(dropna=False)
        self.test_classes = pd.Series(self.test_value_cnts.index.values)
        if self.retain_nan:
            test_in_nan = self.test_classes.isin(self.nan_classes)
            self.test_classes = self.test_classes[~test_in_nan]
        invert_s = pd.Series(index=self.train_classes, 
                             data=self.train_classes.index.values, 
                             dtype=np.object)
        inds_s = y_test.map(invert_s)
        if self.retain_nan:
            inds_s = self._ind_to_nan(inds_s, y_test)
        test_in_train = self.test_classes.isin(self.train_classes)
        diff = self.test_classes[~test_in_train]
        if list(diff):
            print "y contains new labels!"
            self.unknown_classes = diff
            is_unknown_cls = y_test.isin(self.unknown_classes)
            if self.use_mode:
                is_mode = self.train_classes == self.most_common_class.values[0]
                inds_s[is_unknown_cls] = self.most_common_class.index.values[0]
            else:
                inds_s[is_unknown_cls] = self.train_classes.index[-1]+1
            if self.drop_unknown_classes:
                inds_s = inds_s[~is_unknown_cls]
        if self.retain_nan and np.isnan(self.nan_classes).all():
            inds_s = inds_s.astype(np.float64)
        elif not self.retain_nan:
            inds_s = inds_s.astype(np.int64)
        return inds_s.values

    def _ind_to_nan(self, indcs, y):
        l = list(self.nan_classes)
        if np.nan in l:
            indcs[pd.isnull(y)] = np.nan
            l.remove(np.nan)
        for nanclass in l:
            indcs[y==nanclass] = nanclass
        return indcs    
    
    def inverse_transform(self, indcs):
        #is y a numpy array?
        assert type(indcs).__module__ == np.__name__
        check_is_fitted(self, 'train_classes')
        indcs = pd.Series(indcs)
        is_in_range = indcs.isin(self.train_classes.index)
        is_out_of_range = np.invert(is_in_range)
        if is_out_of_range.any():
            print "vector contains unknown indices!"
            z = np.empty(np.size(indcs), dtype=np.object)
            z[is_in_range] = self.train_classes[indcs[is_in_range]]
            z[is_out_of_range] = "unknown"
            if self.retain_nan==True:
                z = self._ind_to_nan(z, indcs)
            return z
        else:
            return self.train_classes[indcs]


        
        