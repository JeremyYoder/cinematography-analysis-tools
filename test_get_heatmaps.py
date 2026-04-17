import sys
import unittest
from unittest.mock import patch
import importlib.util

class TestGetHeatmaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load get-heatmaps.py
        spec = importlib.util.spec_from_file_location("get_heatmaps", "get-heatmaps.py")
        cls.get_heatmaps = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.get_heatmaps)

    @patch('cinematography_tools.heatmap.generate_heatmaps')
    @patch('sys.argv', ['get-heatmaps.py', '--path_img', '/fake/img/path'])
    def test_main_runs_heatmaps(self, mock_generate_heatmaps):
        self.get_heatmaps.main()

        mock_generate_heatmaps.assert_called_once_with(
            path_base=None,
            path_img='/fake/img/path',
            path_hms=None,
            alpha=0.5
        )

    @patch('cinematography_tools.heatmap.generate_heatmaps')
    @patch('sys.argv', ['get-heatmaps.py', '--path_base', '/my/base', '--path_img', '/fake/img/path', '--path_hms', '/my/hms', '--alpha', '0.8'])
    def test_main_runs_heatmaps_with_all_args(self, mock_generate_heatmaps):
        self.get_heatmaps.main()

        mock_generate_heatmaps.assert_called_once_with(
            path_base='/my/base',
            path_img='/fake/img/path',
            path_hms='/my/hms',
            alpha=0.8
        )

if __name__ == '__main__':
    unittest.main()
