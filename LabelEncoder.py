import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted
import pandas as pd



class LabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, nan_classes=[np.nan], retain_nan=False, drop_unknown_classes=False, use_mode=True):
        self.unknown_classes = []
        self.classes = []
        self.nan_classes = nan_classes
        self.retain_nan = retain_nan
        self.drop_unknown_classes = drop_unknown_classes
        self.use_mode =  use_mode
        self.test_classes = []
        
        
    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def fit(self, y_train):
        y_train = pd.Series(y_train, dtype=np.object)
        val_cnts = y_train.value_counts()
        self.most_common_class = val_cnts.index[0]
        self.classes = pd.Series(data = val_cnts.index)
        if self.retain_nan:
            self.classes = self.classes[np.invert(self.classes.isin(self.nan_classes))]
        

    def transform(self, y_test):
        check_is_fitted(self, 'classes')
        y_test = pd.Series(y_test, dtype=np.object)
        self.test_classes = pd.Series(y_test.unique())
        if self.retain_nan:
            self.test_classes = self.test_classes[np.invert(self.test_classes.isin(self.nan_classes))]
        invert_s = pd.Series(index=self.classes, 
                             data=self.classes.index.values, dtype=np.object)
        inds_s = y_test.map(invert_s)
        
        if self.retain_nan:
            inds_s = self._ind_to_nan(inds_s, y_test)
        
        diff = np.setdiff1d(self.test_classes, self.classes) #set(test_classes) - set(self.classes)
        if list(diff):
            warnings.warn("y contains new labels!")
            self.unknown_classes = diff
            is_unknown_cls = y_test.isin(self.unknown_classes)
            if pd.isnull(self.unknown_classes).any():
                is_nan_cls = y_test.isnull()
                is_unknown_cls = is_unknown_cls | is_nan_cls
            if self.use_mode:
                inds_s[is_unknown_cls] = 0
            else:
                inds_s[is_unknown_cls] = len(self.classes)
            if self.drop_unknown_classes:
                inds_s = inds_s[np.invert(is_unknown_cls)]
        
        if not self.retain_nan:
            inds_s = inds_s.astype(np.int64)
            
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
        l = self.nan_classes
        if np.nan in l:
            indcs[pd.isnull(y)] = np.nan
            l.remove(np.nan)
        for nanclass in l:
            indcs[y==nanclass] = nanclass
        return indcs    
                


        
        