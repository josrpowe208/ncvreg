from abc import ABC, abstractmethod, ABCMeta


class BaseRegressor(ABC, ABCMeta):
    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def fit_predict(self):
        pass

    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def fit_transform(self):
        pass
