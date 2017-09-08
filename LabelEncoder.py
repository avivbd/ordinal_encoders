import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted
import pandas as pd



class LabelEncoder(BaseEstimator, TransformerMixin):
    """Encode labels with value between 0 and n_classes-1.

    Read more in the :ref:`User Guide <preprocessing_targets>`.
    Modified from sklearn by AB Aug 28 2017

    Attributes
    ----------
    classes_ : array of shape (n_class,)
        Holds the label for each class.

    Examples
    --------
    >>> le = LabelEncoder()
    >>> y_train = ["paris", "paris", "tokyo"]
    >>> le.fit(y_train)
    >>> le.classes_
    array(['paris', 'tokyo'],dtype='|S8')

    >>> le.transform(y_train)
    array([0, 0, 1])

    >>> y_test = ["tokyo", "tokyo", "paris", "Amsterdam"]
    >>> le.transform(y_test)

    UserWarning: y contains new labels! 
    array([ 0,  0,  1, -1])

    Note that unknown classes are mapped to n+1 where n is the size of the training dictionary.
    The unknown classes are saved in:

    >>> le.unknown_classes
    array(['Amsterdam'], dtype='|S9')

    >>> le.inverse_transform(np.array([0, 0, 1, -1, 0, 2]))
    array(['paris', 'paris', 'tokyo', 'unknown', 'paris', 'unknown'], dtype=object)

    See also
    --------
    sklearn.preprocessing.OneHotEncoder : encode categorical integer features
        using a one-hot aka one-of-K scheme.
    """

    def __init__(self):
	self.unknown_classes = []
	self.classes_ = []

    def fit(self, y):
        """Fit label encoder

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        y = column_or_1d(y, warn=True)
        self.classes_ = np.unique(y)
        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        y = column_or_1d(y, warn=True)
        self.classes_, y = np.unique(y, return_inverse=True)

        return y

    def transform(self, y, drop=False):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """
        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        classes = np.unique(y)
        diff = np.setdiff1d(classes, self.classes_)
        if list(diff):
            self.unknown_classes = diff
            warnings.warn("y contains new labels! See LabelEncoder.unknown_classes")
            is_unknown_cls = np.in1d(y, self.unknown_classes) 
            inds_ar = np.searchsorted(self.classes_, y)
            inds_ar[is_unknown_cls] = len(self.classes_)
            if drop==True:
                return inds_ar[np.invert(is_unknown_cls)]
            else:
                return inds_ar
            
        else:
            return np.searchsorted(self.classes_, y)

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        #is y a numpy array?
        assert type(y).__module__ == np.__name__
        
        check_is_fitted(self, 'classes_')
        out_of_range_values = np.setdiff1d(y, range(len(self.classes_)))
        is_out_of_range = np.in1d(y, out_of_range_values)
        has_out_of_range = np.any(is_out_of_range)
        if has_out_of_range:
            warnings.warn("y contains unknown labels")
            z = np.empty(np.size(y), dtype=np.object)
            is_in_range = np.invert(is_out_of_range)
            z[is_in_range] = self.classes_[y[is_in_range]]
            z[is_out_of_range] = "unknown"
            return z
        else:
            return self.classes_[y]
