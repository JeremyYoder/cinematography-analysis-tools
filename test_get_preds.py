import sys
import unittest
from unittest.mock import patch
import importlib.util

class TestGetPreds(unittest.TestCase):
    @patch('cinematography_tools.predict.run_predictions')
    def test_run_predictions(self, mock_run_predictions):
        test_args = ['get-preds.py', '--path_img', '/fake/img/path']
        with patch.object(sys, 'argv', test_args):
            spec = importlib.util.spec_from_file_location("get_preds", "get-preds.py")
            get_preds = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(get_preds)
            get_preds.main()

        mock_run_predictions.assert_called_once_with(
            path_base=None,
            path_img='/fake/img/path',
            path_preds=None,
        )

    @patch('cinematography_tools.predict.run_predictions')
    def test_run_predictions_with_path_preds(self, mock_run_predictions):
        test_args = ['get-preds.py', '--path_base', '/fake/base/path', '--path_img', '/fake/img/path', '--path_preds', '/fake/preds/path']
        with patch.object(sys, 'argv', test_args):
            spec = importlib.util.spec_from_file_location("get_preds", "get-preds.py")
            get_preds = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(get_preds)
            get_preds.main()

        mock_run_predictions.assert_called_once_with(
            path_base='/fake/base/path',
            path_img='/fake/img/path',
            path_preds='/fake/preds/path',
        )

if __name__ == '__main__':
    unittest.main()
