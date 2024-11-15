from abc import ABC, abstractmethod, ABCMeta


class BaseRegressor(ABC, ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit_predict(self, *args, **kwargs):
        pass

    @abstractmethod
    def transform(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit_transform(self, *args, **kwargs):
        pass
