import sys
import unittest
from unittest.mock import patch
import importlib.util

# Ensure the module is imported before patching
import cinematography_tools.heatmap

class TestGetHeatmaps(unittest.TestCase):
    @patch('cinematography_tools.heatmap.generate_heatmaps')
    def test_generate_heatmaps(self, mock_generate_heatmaps):
        test_args = ['get-heatmaps.py', '--path_img', '/fake/img/path']
        with patch.object(sys, 'argv', test_args):
            spec = importlib.util.spec_from_file_location("get_heatmaps", "get-heatmaps.py")
            get_heatmaps = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(get_heatmaps)
            get_heatmaps.main()

        mock_generate_heatmaps.assert_called_once_with(
            path_base=None,
            path_img='/fake/img/path',
            path_hms=None,
            alpha=0.5,
        )

    @patch('cinematography_tools.heatmap.generate_heatmaps')
    def test_generate_heatmaps_with_args(self, mock_generate_heatmaps):
        test_args = ['get-heatmaps.py', '--path_base', '/fake/base/path', '--path_img', '/fake/img/path', '--path_hms', '/fake/hms/path', '--alpha', '0.8']
        with patch.object(sys, 'argv', test_args):
            spec = importlib.util.spec_from_file_location("get_heatmaps", "get-heatmaps.py")
            get_heatmaps = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(get_heatmaps)
            get_heatmaps.main()

        mock_generate_heatmaps.assert_called_once_with(
            path_base='/fake/base/path',
            path_img='/fake/img/path',
            path_hms='/fake/hms/path',
            alpha=0.8,
        )

if __name__ == '__main__':
    unittest.main()
