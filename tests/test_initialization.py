import pytest
import numpy as np
from src.ncvreg.models.ncvreg import NCVREG


class TestNCVReg:
    def test_fit_model(self):
        # Load standard dataset for testing
        X = np.random.randn(1000, 10)
        y = np.random.randn(1000)

        # Initialize the ncvreg class
        model = NCVREG()

        # Fit the model to the dataset
        model.fit(X, y)

        # Assert that the model has been fitted correctly
        assert model.fitted == True

