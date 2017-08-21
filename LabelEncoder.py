import numpy as np
import warnings
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import column_or_1d
from sklearn.utils.validation import check_is_fitted




class LabelEncoder(BaseEstimator, TransformerMixin):
    """Encode labels with value between 0 and n_classes-1.

    Read more in the :ref:`User Guide <preprocessing_targets>`.
    Modified from sklearn by AB Aug 21 2017

    Attributes
    ----------
    classes_ : array of shape (n_class,)
        Holds the label for each class.

    Examples
    --------
    >>> le = LabelEncoder()
    >>> y_train = ["tokyo", "tokyo", "paris", "aberdeen"]
    >>> le.fit(y_train)
    >>> le.classes_
    array(['aberdeen', 'paris', 'tokyo'],dtype='|S8')

    >>> le.transform(y_train)
    array([2, 2, 1, 0])

    >>> y_test = ["tokyo", "tokyo", "paris", "Amsterdam"]
    >>> le.transform(y_test)

    UserWarning: y contains new labels: ['Amsterdam']
    array([2, 2, 1])

    (note that rows with unknown classes are dropped)
    (the unknown classes are saved in:)

    >>> le.unknown_classes
    array(['Amsterdam'], dtype='|S9')

    >>> np.where(y_test==le.unknown_classes)
    (array([3]),)

    See also
    --------
    sklearn.preprocessing.OneHotEncoder : encode categorical integer features
        using a one-hot aka one-of-K scheme.
    """

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

    def transform(self, y):
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
        if len(np.intersect1d(classes, self.classes_)) < len(classes):
            diff = np.setdiff1d(classes, self.classes_)
            self.unknown_classes = diff
            warnings.warn("y contains new labels: %s" % str(diff))
            return np.searchsorted(self.classes_, y[y != diff])
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
        check_is_fitted(self, 'classes_')

        diff = np.setdiff1d(y, np.arange(len(self.classes_)))
        if diff:
            self.unknown_classes = diff
            warnings.warn("y contains new labels: %s" % str(diff))
            return self.classes_[y != diff]
        else:
            return self.classes_[y]