import sys
import unittest
from unittest.mock import patch
import importlib.util

class TestGetPreds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load get-preds.py
        spec = importlib.util.spec_from_file_location("get_preds", "get-preds.py")
        cls.get_preds = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.get_preds)

    @patch('cinematography_tools.predict.run_predictions')
    @patch('sys.argv', ['get-preds.py', '--path_img', '/fake/img/path'])
    def test_main_runs_predictions(self, mock_run_predictions):
        self.get_preds.main()

        mock_run_predictions.assert_called_once_with(
            path_base=None,
            path_img='/fake/img/path',
            path_preds=None
        )

    @patch('cinematography_tools.predict.run_predictions')
    @patch('sys.argv', ['get-preds.py', '--path_base', '/my/base', '--path_img', '/fake/img/path', '--path_preds', '/my/preds'])
    def test_main_runs_predictions_with_all_args(self, mock_run_predictions):
        self.get_preds.main()

        mock_run_predictions.assert_called_once_with(
            path_base='/my/base',
            path_img='/fake/img/path',
            path_preds='/my/preds'
        )

if __name__ == '__main__':
    unittest.main()
