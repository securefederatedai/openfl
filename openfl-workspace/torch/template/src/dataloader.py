# Copyright (C) 2024 Intel Corporation
# Licensed subject to the terms of the separately executed evaluation license agreement between
# Intel Corporation and you.

from openfl.federated import PyTorchDataLoader


class TemplateDataLoader(PyTorchDataLoader):
    """Template dataloader for PyTorch.
    This class should be used as a template to create a custom DataLoader for your specific dataset.
    After generating this template, you should:
    1. Implement the `load_dataset` function to load your data.
    2. Modify the `plan.yaml` file to use this DataLoader.
    The `plan.yaml` modifications should be done under the `<workspace>/plan/plan.yaml` section:
    ```
    data_loader:
        defaults: plan/defaults/data_loader.yaml
        template: src.dataloader.TemplateDataLoader # Modify this line appropriately if you
                  change the class name
        settings:
            # Add additional arguments (such as batch_size) that you wish
            # to pass through `def __init__():`
            # You do not need to pass in data_path here. It will be set by the collaborators
    ```
    `batch_size` is passed to the `super().`__init__` method to ensure that the superclass
    is properly initialized with the specified batch size.
    After calling `super().__init__`, define `self.X_train`, `self.y_train`,
    `self.X_valid`, and `self.y_valid`.

    Note: If data_path is None, the dataloader is being initialized for model creation only,
    and no data will be loaded.
    """

    def __init__(self, data_path=None, batch_size=32, **kwargs):
        """Initialize the data loader.
        Args:
            data_path: The file path to the data at the respective collaborator site.
                       If None, initialize for model creation only.
            batch_size: The batch size of the data loader.
            **kwargs: Additional arguments that may be defined in `plan.yaml`
        """
        super().__init__(batch_size, **kwargs)

        # Define required attributes for minimal initialization
        # These should be replaced with appropriate values for your specific dataset
        self.feature_shape = [1, 28, 28]  # Example shape [channels, height, width] for PyTorch
        self.num_classes = 10  # Example number of classes

        # If data_path is None, this is being used for model initialization only
        if data_path is None:
            return

        # Load the dataset using the provided data_path and any additional kwargs.
        X_train, y_train, X_valid, y_valid = load_dataset(data_path, **kwargs)

        # Assign the loaded data to instance variables.
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid

    def get_feature_shape(self):
        """Returns the shape of an example feature array.

        Returns:
            list: The shape of an example feature array.
        """
        return self.feature_shape

    def get_num_classes(self):
        """Returns the number of classes for classification tasks.

        Returns:
            int: The number of classes.
        """
        return self.num_classes


def load_dataset(data_path, **kwargs):
    """
    Load your dataset here.
    This function should be implemented to load the dataset from the given `data_path`.
    You can use additional arguments passed via `**kwargs` if necessary.
    Args:
        data_path (str): Path to the data directory.
        **kwargs: Additional arguments that may be defined in `plan.yaml`
    Returns:
        Tuple containing:
        - numpy.ndarray: The training data.
        - numpy.ndarray: The training labels.
        - numpy.ndarray: The validation data.
        - numpy.ndarray: The validation labels.
    """
    # Implement dataset loading logic here and return the appropriate data.
    # Replace the following placeholders with actual data loading code.
    X_train = None  # Placeholder for training data.
    y_train = None  # Placeholder for training labels.
    X_valid = None  # Placeholder for validation data.
    y_valid = None  # Placeholder for validation labels.

    return X_train, y_train, X_valid, y_valid


raise NotImplementedError("Use <workspace>/src/dataloader.py template to "
                          "create a custom dataloader. Then remove this line.")
