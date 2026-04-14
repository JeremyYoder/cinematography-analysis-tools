import sys
import unittest
from unittest.mock import patch
import importlib.util

spec = importlib.util.spec_from_file_location("get_preds", "get-preds.py")
get_preds = importlib.util.module_from_spec(spec)
sys.modules["get_preds"] = get_preds
spec.loader.exec_module(get_preds)

class TestGetPreds(unittest.TestCase):
    @patch('cinematography_tools.predict.run_predictions')
    def test_main_parses_args(self, mock_run_predictions):
        test_args = ['get-preds.py', '--path_base', '/base', '--path_img', '/img', '--path_preds', '/preds']
        with patch.object(sys, 'argv', test_args):
            get_preds.main()
            mock_run_predictions.assert_called_once_with(
                path_base='/base',
                path_img='/img',
                path_preds='/preds'
            )

    @patch('cinematography_tools.predict.run_predictions')
    def test_main_default_args(self, mock_run_predictions):
        test_args = ['get-preds.py', '--path_img', '/img']
        with patch.object(sys, 'argv', test_args):
            get_preds.main()
            mock_run_predictions.assert_called_once_with(
                path_base=None,
                path_img='/img',
                path_preds=None
            )

if __name__ == '__main__':
    unittest.main()
