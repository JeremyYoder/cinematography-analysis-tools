import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib.util

spec = importlib.util.spec_from_file_location("get_preds", "get-preds.py")
get_preds = importlib.util.module_from_spec(spec)
sys.modules["get_preds"] = get_preds
spec.loader.exec_module(get_preds)

class TestGetPreds(unittest.TestCase):
    def test_main(self):
        mock_run_predictions = MagicMock()

        with patch.object(sys, 'argv', ['get-preds.py', '--path_base', '/base', '--path_img', '/img', '--path_preds', '/preds']):
            original_import = __import__
            def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == 'cinematography_tools.predict':
                    mock_module = MagicMock()
                    mock_module.run_predictions = mock_run_predictions
                    return mock_module
                return original_import(name, globals, locals, fromlist, level)

            with patch('builtins.__import__', side_effect=mock_import):
                get_preds.main()

        mock_run_predictions.assert_called_once_with(
            path_base='/base',
            path_img='/img',
            path_preds='/preds',
        )

    @patch('builtins.print')
    @patch('sys.exit')
    def test_main_import_error(self, mock_exit, mock_print):
        with patch.object(sys, 'argv', ['get-preds.py', '--path_img', '/img']):
            # Simulate ImportError when importing from cinematography_tools.predict
            original_import = __import__
            def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == 'cinematography_tools.predict':
                    raise ImportError("Mocked import error")
                return original_import(name, globals, locals, fromlist, level)

            with patch('builtins.__import__', side_effect=mock_import):
                get_preds.main()

        mock_print.assert_any_call("Error: cinematography_tools package not installed.")
        mock_exit.assert_called_once_with(1)

if __name__ == '__main__':
    unittest.main()
