import unittest

from sklearn.datasets import fetch_openml, load_breast_cancer
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier, TabPFNRegressor


class TestTabPFNClassification(unittest.TestCase):
    """Test cases for TabPFN classification"""

    def setUp(self):
        """Load data and initialize classifier for all tests"""
        self.X, self.y = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.7, random_state=42
        )
        print(self.X_train.shape)
        self.clf = TabPFNClassifier()
        self.clf.fit(self.X_train, self.y_train)

    def test_predict_proba(self):
        """Test probability predictions"""
        prediction_probabilities = self.clf.predict_proba(self.X_test)
        roc_auc = roc_auc_score(self.y_test, prediction_probabilities[:, 1])
        self.assertGreater(roc_auc, 0.5, "ROC AUC should be greater than 0.5")

    def test_predict(self):
        """Test label predictions"""
        predictions = self.clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        self.assertGreater(accuracy, 0.5, "Accuracy should be greater than 0.5")


class TestTabPFNRegression(unittest.TestCase):
    """Test cases for TabPFN regression"""

    def setUp(self):
        """Load data and initialize regressor for all tests"""
        df = fetch_openml(data_id=531, as_frame=True)  # Boston Housing dataset
        self.X = df.data
        self.y = df.target.astype(float)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.5, random_state=42
        )
        self.regressor = TabPFNRegressor()
        self.regressor.fit(self.X_train, self.y_train)

    def test_regression_metrics(self):
        """Test regression metrics"""
        predictions = self.regressor.predict(self.X_test)
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)

        self.assertLess(mse, 100, "MSE should be reasonably low")
        self.assertGreater(r2, 0.5, "R² score should be greater than 0.5")


# TODO: Add TestTabPFNTimeSeries class when time series functionality is implemented
# class TestTabPFNTimeSeries(unittest.TestCase):
#     """Test cases for TabPFN time series (to be implemented)"""
#     pass

if __name__ == "__main__":
    unittest.main()
