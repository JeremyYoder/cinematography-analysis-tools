import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib.util

spec = importlib.util.spec_from_file_location("get_heatmaps", "get-heatmaps.py")
get_heatmaps = importlib.util.module_from_spec(spec)
sys.modules["get_heatmaps"] = get_heatmaps
spec.loader.exec_module(get_heatmaps)

class TestGetHeatmaps(unittest.TestCase):
    def test_main(self):
        mock_generate_heatmaps = MagicMock()

        with patch.object(sys, 'argv', ['get-heatmaps.py', '--path_base', '/base', '--path_img', '/img', '--path_hms', '/hms', '--alpha', '0.7']):
            original_import = __import__
            def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == 'cinematography_tools.heatmap':
                    mock_module = MagicMock()
                    mock_module.generate_heatmaps = mock_generate_heatmaps
                    return mock_module
                return original_import(name, globals, locals, fromlist, level)

            with patch('builtins.__import__', side_effect=mock_import):
                get_heatmaps.main()

        mock_generate_heatmaps.assert_called_once_with(
            path_base='/base',
            path_img='/img',
            path_hms='/hms',
            alpha=0.7,
        )

    @patch('builtins.print')
    @patch('sys.exit')
    def test_main_import_error(self, mock_exit, mock_print):
        with patch.object(sys, 'argv', ['get-heatmaps.py', '--path_img', '/img']):
            original_import = __import__
            def mock_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == 'cinematography_tools.heatmap':
                    raise ImportError("Mocked import error")
                return original_import(name, globals, locals, fromlist, level)

            with patch('builtins.__import__', side_effect=mock_import):
                get_heatmaps.main()

        mock_print.assert_any_call("Error: cinematography_tools package not installed.")
        mock_exit.assert_called_once_with(1)

if __name__ == '__main__':
    unittest.main()
