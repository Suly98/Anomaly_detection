import unittest
import numpy as np
from anomalies import generate_data, anomalies_detector

class TestAnomaliesScript(unittest.TestCase):
    def test_generate_data_valid(self):
        """Test generate_data to produce the correct # of data"""
        result = generate_data(100)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 100)

    def test_generate_data_invalid_size(self):
        """Test generate_data raises a ValueError for invalid sizes"""
        with self.assertRaises(ValueError):
            generate_data(-10)

    def test_anomalies_detector(self):
        """Test anomalies_detector correctly identifies anomalies."""
        test_data = np.concatenate((np.random.normal(0, 1, 100), np.random.uniform(10, 15, 10)))
        predictions = anomalies_detector(test_data)
        self.assertEqual(len(predictions), len(test_data))

        #check if some anomalies were detected!
        self.assertIn(-1, predictions)
