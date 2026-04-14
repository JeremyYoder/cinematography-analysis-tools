import sys
import unittest
from unittest.mock import patch
import importlib.util

spec = importlib.util.spec_from_file_location("get_heatmaps", "get-heatmaps.py")
get_heatmaps = importlib.util.module_from_spec(spec)
sys.modules["get_heatmaps"] = get_heatmaps
spec.loader.exec_module(get_heatmaps)

class TestGetHeatmaps(unittest.TestCase):
    @patch('cinematography_tools.heatmap.generate_heatmaps')
    def test_main_parses_args(self, mock_generate_heatmaps):
        test_args = ['get-heatmaps.py', '--path_base', '/base', '--path_img', '/img', '--path_hms', '/hms', '--alpha', '0.8']
        with patch.object(sys, 'argv', test_args):
            get_heatmaps.main()
            mock_generate_heatmaps.assert_called_once_with(
                path_base='/base',
                path_img='/img',
                path_hms='/hms',
                alpha=0.8
            )

    @patch('cinematography_tools.heatmap.generate_heatmaps')
    def test_main_default_args(self, mock_generate_heatmaps):
        test_args = ['get-heatmaps.py', '--path_img', '/img']
        with patch.object(sys, 'argv', test_args):
            get_heatmaps.main()
            mock_generate_heatmaps.assert_called_once_with(
                path_base=None,
                path_img='/img',
                path_hms=None,
                alpha=0.5
            )

if __name__ == '__main__':
    unittest.main()
