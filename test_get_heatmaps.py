import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib

class TestGetHeatmaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_modules = {
            'fastai': sys.modules.get('fastai'),
            'fastai.vision': sys.modules.get('fastai.vision'),
            'fastai.callbacks': sys.modules.get('fastai.callbacks'),
            'fastai.callbacks.hooks': sys.modules.get('fastai.callbacks.hooks'),
            'torch': sys.modules.get('torch'),
            'matplotlib': sys.modules.get('matplotlib'),
            'matplotlib.pyplot': sys.modules.get('matplotlib.pyplot'),
            'matplotlib.ticker': sys.modules.get('matplotlib.ticker'),
        }

        sys.modules['fastai'] = MagicMock()
        sys.modules['fastai.vision'] = MagicMock()
        sys.modules['fastai.callbacks'] = MagicMock()
        sys.modules['fastai.callbacks.hooks'] = MagicMock()
        sys.modules['torch'] = MagicMock()
        sys.modules['matplotlib'] = MagicMock()
        sys.modules['matplotlib.pyplot'] = MagicMock()
        sys.modules['matplotlib.ticker'] = MagicMock()

        global get_heatmaps
        get_heatmaps = importlib.import_module('get-heatmaps')

    @classmethod
    def tearDownClass(cls):
        for name, module in cls.original_modules.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module

    @patch('os.listdir')
    @patch('pathlib.Path.exists')
    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('os.rename')
    def test_generate_heatmaps_no_images(self, mock_rename, mock_mkdir, mock_exists, mock_path_exists, mock_listdir):
        # Empty directory
        mock_listdir.return_value = []
        mock_exists.return_value = True
        mock_path_exists.return_value = True

        with patch.object(get_heatmaps, 'rmtree') as mock_rmtree:
            get_heatmaps.generate_heatmaps('/mock/img/dir')
            mock_listdir.assert_called_with(unittest.mock.ANY)

    @patch('os.listdir')
    @patch('pathlib.Path.exists')
    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('os.rename')
    def test_generate_heatmaps(self, mock_rename, mock_mkdir, mock_exists, mock_path_exists, mock_listdir):
        with patch.object(get_heatmaps.ImageDataBunch, 'from_folder') as mock_from_folder, \
             patch.object(get_heatmaps, 'hooked_backward') as mock_hook, \
             patch.object(get_heatmaps, 'save_img') as mock_save, \
             patch.object(get_heatmaps, 'show_heatmap') as mock_show, \
             patch.object(get_heatmaps, 'Image') as mock_image, \
             patch.object(get_heatmaps, 'rmtree') as mock_rmtree:

            mock_listdir.return_value = ['img1.jpg']
            mock_exists.return_value = False # Let it create directories
            mock_path_exists.return_value = True

            mock_temp = MagicMock()
            mock_temp.train_ds = [(MagicMock(), MagicMock())]
            mock_temp.one_item.return_value = [MagicMock()]
            mock_temp.denorm.return_value = [MagicMock()]
            mock_from_folder.return_value.normalize.return_value = mock_temp

            mock_learn = MagicMock()
            mock_model = MagicMock()
            mock_learn.model.eval.return_value = mock_model

            mock_hook_a = MagicMock()
            mock_hook_g = MagicMock()
            mock_hook_a.stored = [MagicMock(cpu=MagicMock(return_value=MagicMock(mean=MagicMock(return_value=MagicMock()))))]
            mock_hook.return_value = (mock_hook_a, mock_hook_g)

            get_heatmaps.generate_heatmaps('/mock/img/dir', '/mock/hms/dir', 0.8, mock_learn)

            mock_mkdir.assert_called()
            mock_rename.assert_called()
            self.assertEqual(mock_rename.call_count, 2) # Move in, move out
            mock_rmtree.assert_called()

            mock_hook.assert_called_once()
            mock_save.assert_called_once()
            mock_show.assert_called_once()
            args, kwargs = mock_show.call_args
            self.assertEqual(kwargs['alpha'], 0.8)

if __name__ == '__main__':
    unittest.main()
