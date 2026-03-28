import sys
import unittest
from unittest.mock import MagicMock, patch
import os
import importlib

# Mock heavy dependencies globally before importing get-heatmaps
mock_torch = MagicMock()
mock_matplotlib = MagicMock()
mock_matplotlib_pyplot = MagicMock()
mock_matplotlib_ticker = MagicMock()
mock_fastai = MagicMock()
mock_fastai_vision = MagicMock()
mock_fastai_callbacks = MagicMock()

sys.modules['torch'] = mock_torch
sys.modules['matplotlib'] = mock_matplotlib
sys.modules['matplotlib.pyplot'] = mock_matplotlib_pyplot
sys.modules['matplotlib.ticker'] = mock_matplotlib_ticker
sys.modules['fastai'] = mock_fastai
sys.modules['fastai.vision'] = mock_fastai_vision
sys.modules['fastai.callbacks'] = mock_fastai_callbacks
sys.modules['fastai.callbacks.hooks'] = mock_fastai_callbacks.hooks

class TestGetHeatmaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We use importlib because of the hyphen in get-heatmaps.py
        cls.get_heatmaps = importlib.import_module('get-heatmaps')

    def test_show_heatmap(self):
        with patch.object(self.get_heatmaps.plt, 'subplots') as mock_subplots, \
             patch.object(self.get_heatmaps.plt, 'gca') as mock_gca, \
             patch.object(self.get_heatmaps.plt, 'savefig') as mock_savefig, \
             patch.object(self.get_heatmaps.plt, 'close') as mock_close:

            mock_ax = MagicMock()
            mock_subplots.return_value = (None, mock_ax)

            mock_hm = MagicMock()
            mock_path = MagicMock()
            mock_path.__truediv__.return_value = 'fake_path/LS_1_heatmap.png' # for path / fname

            mock_xb_im = MagicMock()

            self.get_heatmaps.show_heatmap(mock_hm, mock_path, 'LS', 0, mock_xb_im)

            # verify xb_im.show was called
            mock_xb_im.show.assert_called_with(mock_ax)

            # verify ax.imshow was called
            mock_ax.imshow.assert_called()

            # verify savefig was called
            mock_savefig.assert_called_with('fake_path/LS_1_heatmap.png', bbox_inches='tight', pad_inches=0, dpi=800)

            # verify plt.close was called
            self.assertTrue(mock_close.called)

    def test_save_img(self):
        with patch.object(self.get_heatmaps.plt, 'gca') as mock_gca, \
             patch.object(self.get_heatmaps.plt, 'savefig') as mock_savefig, \
             patch.object(self.get_heatmaps.plt, 'close') as mock_close:

            mock_img = MagicMock()
            mock_path = MagicMock()
            mock_path.__truediv__.return_value = 'fake_path/LS_1.png'

            self.get_heatmaps.save_img(mock_img, mock_path, 'LS', 0)

            mock_img.show.assert_called_with(figsize=(5,3))
            mock_savefig.assert_called_with('fake_path/LS_1.png', bbox_inches='tight', pad_inches=0, dpi=800)
            self.assertTrue(mock_close.called)

if __name__ == '__main__':
    unittest.main()
