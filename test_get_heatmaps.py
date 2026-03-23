import sys
import unittest
from unittest.mock import MagicMock, patch, ANY
import importlib

class TestGetHeatmaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock heavy dependencies globally for this test suite
        cls.mock_fastai = MagicMock()
        cls.mock_fastai_callbacks_hooks = MagicMock()
        cls.mock_fastai_vision = MagicMock()
        cls.mock_matplotlib = MagicMock()
        cls.mock_matplotlib_pyplot = MagicMock()
        cls.mock_matplotlib_ticker = MagicMock()
        cls.mock_torch = MagicMock()

        # Save original modules to restore later if needed
        cls.original_modules = {
            'fastai': sys.modules.get('fastai'),
            'fastai.callbacks.hooks': sys.modules.get('fastai.callbacks.hooks'),
            'fastai.vision': sys.modules.get('fastai.vision'),
            'matplotlib': sys.modules.get('matplotlib'),
            'matplotlib.pyplot': sys.modules.get('matplotlib.pyplot'),
            'matplotlib.ticker': sys.modules.get('matplotlib.ticker'),
            'torch': sys.modules.get('torch'),
        }

        sys.modules['fastai'] = cls.mock_fastai
        sys.modules['fastai.callbacks.hooks'] = cls.mock_fastai_callbacks_hooks
        sys.modules['fastai.vision'] = cls.mock_fastai_vision
        sys.modules['matplotlib'] = cls.mock_matplotlib
        sys.modules['matplotlib.pyplot'] = cls.mock_matplotlib_pyplot
        sys.modules['matplotlib.ticker'] = cls.mock_matplotlib_ticker
        sys.modules['torch'] = cls.mock_torch

        from functools import partial
        cls.mock_fastai_vision.partial = partial

        # Mocking specific functions and classes in fastai.vision that initialise.py expects
        for func in ['cutout', 'jitter', 'skew', 'squish', 'tilt',
                     'perspective_warp', 'crop_pad', 'rgb_randomize',
                     'get_transforms', 'ImageDataBunch', 'ResizeMethod',
                     'imagenet_stats', 'cnn_learner', 'models', 'accuracy', 'open_image', 'Path', 'Image']:
            setattr(cls.mock_fastai_vision, func, MagicMock(name=func))

        cls.mock_fastai_callbacks_hooks.hook_output = MagicMock()
        cls.mock_matplotlib_pyplot.subplots.return_value = (MagicMock(), MagicMock())

        # Import the script
        cls.get_heatmaps = importlib.import_module('get-heatmaps')

    @classmethod
    def tearDownClass(cls):
        # Restore sys.modules
        for mod, original in cls.original_modules.items():
            if original is None:
                del sys.modules[mod]
            else:
                sys.modules[mod] = original

    @patch('os.listdir')
    @patch('os.rename')
    @patch('os.mkdir')
    @patch('os.path.exists')
    def test_generate_heatmaps_creates_directories_and_processes(self, mock_exists, mock_mkdir, mock_rename, mock_listdir):
        # Setup mocks
        mock_listdir.return_value = ['img1.jpg', 'img2.png']
        mock_exists.return_value = False

        mock_learn = MagicMock()
        mock_data = MagicMock()

        # Mock ImageDataBunch creation inside the function
        mock_temp = MagicMock()
        mock_temp.train_ds = [(MagicMock(), 0), (MagicMock(), 1)]
        mock_temp.one_item.return_value = [MagicMock()]
        mock_temp.denorm.return_value = [MagicMock()]
        # Note: the test runner might clear globals differently, so re-apply the mock
        self.get_heatmaps.ImageDataBunch.from_folder.return_value.normalize.return_value = mock_temp

        # Mock hooks
        mock_hook_a = MagicMock()
        mock_acts = MagicMock()
        mock_hook_a.stored = [mock_acts]
        mock_hook_g = MagicMock()

        # Mock Image class inside get_heatmaps
        self.get_heatmaps.Image = MagicMock()

        # We need to patch hooked_backward, show_heatmap, save_img to avoid running real logic and check they were called
        with patch.object(self.get_heatmaps, 'hooked_backward', return_value=(mock_hook_a, mock_hook_g)) as mock_hooked_backward, \
             patch.object(self.get_heatmaps, 'show_heatmap') as mock_show_heatmap, \
             patch.object(self.get_heatmaps, 'save_img') as mock_save_img:

            with patch.object(self.get_heatmaps, 'rmtree') as mock_rmtree:
                self.get_heatmaps.generate_heatmaps(mock_learn, mock_data, '/fake/img/path', '/fake/hms/path')

                # Verify dir operations
                # The code calls os.mkdir(path_hms) where path_hms is Path('/fake/hms/path')
                self.assertTrue(mock_mkdir.called)

                # Verify it processed both images
                self.assertEqual(mock_hooked_backward.call_count, 2)
                self.assertEqual(mock_show_heatmap.call_count, 2)
                self.assertEqual(mock_save_img.call_count, 2)

                # Verify cleanup
                mock_rmtree.assert_called_once()

if __name__ == '__main__':
    unittest.main()
