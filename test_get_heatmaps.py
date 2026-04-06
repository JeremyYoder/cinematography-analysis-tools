import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib.util
from pathlib import Path

class TestGetHeatmaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create global mocks
        cls.mock_fastai = MagicMock()
        cls.mock_fastai_vision = MagicMock()
        cls.mock_fastai_callbacks = MagicMock()
        cls.mock_fastai_callbacks_hooks = MagicMock()
        cls.mock_matplotlib = MagicMock()
        cls.mock_matplotlib_pyplot = MagicMock()
        cls.mock_matplotlib_ticker = MagicMock()
        cls.mock_torch = MagicMock()

        # Inject them into sys.modules
        cls.original_modules = dict(sys.modules)

        sys.modules['fastai'] = cls.mock_fastai
        sys.modules['fastai.vision'] = cls.mock_fastai_vision
        sys.modules['fastai.callbacks'] = cls.mock_fastai_callbacks
        sys.modules['fastai.callbacks.hooks'] = cls.mock_fastai_callbacks_hooks
        sys.modules['matplotlib'] = cls.mock_matplotlib
        sys.modules['matplotlib.pyplot'] = cls.mock_matplotlib_pyplot
        sys.modules['matplotlib.ticker'] = cls.mock_matplotlib_ticker
        sys.modules['torch'] = cls.mock_torch

        # Mock functions for initialise.py
        for func in ['Image', 'ImageDataBunch', 'ResizeMethod', 'imagenet_stats', 'get_model_data', 'get_transforms', 'models', 'accuracy', 'cnn_learner', 'cutout', 'jitter', 'skew', 'squish', 'tilt', 'perspective_warp', 'crop_pad', 'rgb_randomize']:
            setattr(cls.mock_fastai_vision, func, MagicMock(name=func))

        cls.mock_fastai_callbacks_hooks.hook_output = MagicMock(name='hook_output')

        from functools import partial
        cls.mock_fastai_vision.partial = partial

        # Load get_heatmaps
        spec = importlib.util.spec_from_file_location("get_heatmaps", "get-heatmaps.py")
        cls.get_heatmaps = importlib.util.module_from_spec(spec)
        sys.modules["get_heatmaps"] = cls.get_heatmaps
        spec.loader.exec_module(cls.get_heatmaps)

    def test_hooked_backward(self):
        # Mock inputs
        mock_m = MagicMock()
        mock_xb = MagicMock()
        mock_y = 1

        mock_preds = MagicMock()
        mock_m.return_value = mock_preds

        # Mock hook_output context managers
        mock_hook_a_instance = MagicMock()
        mock_hook_g_instance = MagicMock()

        self.mock_fastai_callbacks_hooks.hook_output.side_effect = [
            MagicMock(__enter__=MagicMock(return_value=mock_hook_a_instance)),
            MagicMock(__enter__=MagicMock(return_value=mock_hook_g_instance))
        ]

        # Call the function
        hook_a, hook_g = self.get_heatmaps.hooked_backward(mock_m, mock_xb, mock_y)

        # Verify outputs
        self.assertEqual(hook_a, mock_hook_a_instance)
        self.assertEqual(hook_g, mock_hook_g_instance)

        # Verify hook_output was called twice with m[0]
        self.assertEqual(self.mock_fastai_callbacks_hooks.hook_output.call_count, 2)
        self.mock_fastai_callbacks_hooks.hook_output.assert_any_call(mock_m[0])
        self.mock_fastai_callbacks_hooks.hook_output.assert_any_call(mock_m[0], grad=True)

        # Verify the model was called with xb
        mock_m.assert_called_once_with(mock_xb)

        # Verify backward was called on the correct prediction element
        mock_preds.__getitem__.assert_called_with((0, 1))
        mock_preds.__getitem__.return_value.backward.assert_called_once()

    @patch('get_heatmaps.plt')
    def test_show_heatmap(self, mock_plt):
        # Mocks
        mock_xb_im = MagicMock()
        mock_hm = MagicMock()
        path = Path('/fake/path')
        y = 2
        idx = 5

        # Setup subplot return value
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (None, mock_ax)
        mock_plt.gca.return_value = MagicMock()

        # Call with only_heatmap=False
        self.get_heatmaps.show_heatmap(mock_xb_im, mock_hm, path, y, idx, only_heatmap=False)

        # Verify
        mock_plt.subplots.assert_called_once_with(figsize=(5,3))
        mock_xb_im.show.assert_called_once_with(mock_ax)
        mock_ax.imshow.assert_called_once_with(mock_hm, alpha=0.5, extent=(0,666,375,0),
                                              interpolation='bilinear', cmap='YlOrRd')

        expected_fname = f'{y}_{idx+1}_heatmap.png'
        mock_plt.savefig.assert_called_once_with(path/expected_fname, bbox_inches='tight', pad_inches=0, dpi=800)

        mock_plt.close.assert_any_call()
        mock_plt.close.assert_any_call('all')

    @patch('get_heatmaps.plt')
    def test_save_img(self, mock_plt):
        mock_img = MagicMock()
        path = Path('/fake/path')
        y = 3
        idx = 7

        # Setup mock gca
        mock_gca = MagicMock()
        mock_plt.gca.return_value = mock_gca

        self.get_heatmaps.save_img(mock_img, path, y, idx)

        # Verify
        mock_img.show.assert_called_once_with(figsize=(5,3))
        mock_gca.set_axis_off.assert_called_once()

        expected_fname = f'{y}_{idx+1}.png'
        mock_plt.savefig.assert_called_once_with(path/expected_fname, bbox_inches='tight', pad_inches=0, dpi=800)

        mock_plt.close.assert_any_call()
        mock_plt.close.assert_any_call('all')

    @classmethod
    def tearDownClass(cls):
        sys.modules.clear()
        sys.modules.update(cls.original_modules)

if __name__ == '__main__':
    unittest.main()
