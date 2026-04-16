import sys
import unittest
from unittest.mock import patch
import importlib.util

# Load the legacy script as a module
spec = importlib.util.spec_from_file_location("get_preds", "get-preds.py")
get_preds = importlib.util.module_from_spec(spec)
sys.modules["get_preds"] = get_preds
spec.loader.exec_module(get_preds)

class TestGetPreds(unittest.TestCase):
    @patch('cinematography_tools.predict.run_predictions')
    def test_main_with_all_args(self, mock_run_predictions):
        test_args = [
            'get-preds.py',
            '--path_base', '/fake/base',
            '--path_img', '/fake/img',
            '--path_preds', '/fake/preds'
        ]
        with patch('sys.argv', test_args):
            get_preds.main()

        mock_run_predictions.assert_called_once_with(
            path_base='/fake/base',
            path_img='/fake/img',
            path_preds='/fake/preds'
        )

    @patch('cinematography_tools.predict.run_predictions')
    def test_main_with_required_args(self, mock_run_predictions):
        test_args = [
            'get-preds.py',
            '--path_img', '/fake/img'
        ]
        with patch('sys.argv', test_args):
            get_preds.main()

        mock_run_predictions.assert_called_once_with(
            path_base=None,
            path_img='/fake/img',
            path_preds=None
        )

if __name__ == '__main__':
    unittest.main()
