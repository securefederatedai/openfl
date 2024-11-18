import xgboost as xgb

from openfl.federated.data.loader import DataLoader


class XGBoostDataLoader(DataLoader):
    """A class used to represent a Data Loader for XGBoost models.

    Attributes:
        batch_size (int): Size of batches used for all data loaders.
        X_train (np.array): Training features.
        y_train (np.array): Training labels.
        X_valid (np.array): Validation features.
        y_valid (np.array): Validation labels.
        random_seed (int, optional): Random seed for data shuffling.
    """

    def __init__(self, batch_size=None, random_seed=None, **kwargs):
        """Initializes the XGBoostDataLoader object with the batch size, random
        seed, and any additional arguments.

        Args:
            batch_size (int): The size of batches used for all data loaders.
            random_seed (int, optional): Random seed for data shuffling.
            kwargs: Additional arguments to pass to the function.
        """
        self.batch_size = batch_size
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.random_seed = random_seed

        # Child classes should have init signature:
        # (self, batch_size, **kwargs), should call this __init__ and then
        # define self.X_train, self.y_train, self.X_valid, and self.y_valid

    def get_feature_shape(self):
        """Returns the shape of an example feature array.

        Returns:
            tuple: The shape of an example feature array.
        """
        return self.X_train[0].shape

    def get_train_data_size(self):
        """Returns the total number of training samples.

        Returns:
            int: The total number of training samples.
        """
        return self.X_train.shape[0]

    def get_valid_data_size(self):
        """Returns the total number of validation samples.

        Returns:
            int: The total number of validation samples.
        """
        return self.X_valid.shape[0]

    def get_dmatrix(self, X, y):
        """Returns the DMatrix for the given data.

        Args:
            X (np.array): The input data.
            y (np.array): The label data.

        Returns:
            xgb.DMatrix: The DMatrix object for the given data.
        """
        return xgb.DMatrix(data=X, label=y)

    def get_train_dmatrix(self):
        """Returns the DMatrix for the training data.

        Returns:
            xgb.DMatrix: The DMatrix object for the training data.
        """
        return {"dmatrix": self.get_dmatrix(self.X_train, self.y_train), "labels": self.y_train}

    def get_valid_dmatrix(self):
        """Returns the DMatrix for the validation data.

        Returns:
            xgb.DMatrix: The DMatrix object for the validation data.
        """
        return {"dmatrix": self.get_dmatrix(self.X_valid, self.y_valid), "labels": self.y_valid}
