import sys
import unittest
from unittest.mock import MagicMock, patch
import os
import importlib.util

class TestGetHeatmaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We need to temporarily mock heavy dependencies just for importing get-heatmaps.py
        # and then clean up to avoid polluting other tests
        cls.original_modules = dict(sys.modules)

        mock_fastai = MagicMock()
        mock_fastai_vision = MagicMock()
        mock_fastai_callbacks = MagicMock()
        mock_fastai_callbacks_hooks = MagicMock()
        mock_matplotlib = MagicMock()
        mock_matplotlib_ticker = MagicMock()
        mock_matplotlib_pyplot = MagicMock()
        mock_matplotlib_pyplot.subplots.return_value = (MagicMock(), MagicMock())

        sys.modules['fastai'] = mock_fastai
        sys.modules['fastai.vision'] = mock_fastai_vision
        sys.modules['fastai.callbacks'] = mock_fastai_callbacks
        sys.modules['fastai.callbacks.hooks'] = mock_fastai_callbacks_hooks
        sys.modules['matplotlib'] = mock_matplotlib
        sys.modules['matplotlib.ticker'] = mock_matplotlib_ticker
        sys.modules['matplotlib.pyplot'] = mock_matplotlib_pyplot

        # Load get-heatmaps.py
        spec = importlib.util.spec_from_file_location("get_heatmaps", "get-heatmaps.py")
        cls.get_heatmaps = importlib.util.module_from_spec(spec)
        sys.modules["get_heatmaps"] = cls.get_heatmaps

        cls.get_heatmaps.Path = MagicMock()
        cls.get_heatmaps.os = MagicMock()
        cls.get_heatmaps.rmtree = MagicMock()
        cls.get_heatmaps.get_model_data = MagicMock()
        cls.get_heatmaps.ImageDataBunch = MagicMock()
        cls.get_heatmaps.ResizeMethod = MagicMock()
        cls.get_heatmaps.imagenet_stats = MagicMock()
        cls.get_heatmaps.Image = MagicMock()
        cls.get_heatmaps.torch = MagicMock()

        spec.loader.exec_module(cls.get_heatmaps)

    @classmethod
    def tearDownClass(cls):
        # Restore original sys.modules to prevent global state pollution
        to_remove = [k for k in sys.modules if k not in cls.original_modules]
        for k in to_remove:
            del sys.modules[k]
        sys.modules.update(cls.original_modules)

    def test_generate_heatmaps(self):
        with patch.object(self.get_heatmaps.os, 'path') as mock_path_mod, \
             patch.object(self.get_heatmaps.os, 'listdir') as mock_listdir, \
             patch.object(self.get_heatmaps.os, 'mkdir') as mock_mkdir, \
             patch.object(self.get_heatmaps.os, 'rename') as mock_rename, \
             patch.object(self.get_heatmaps, 'rmtree') as mock_rmtree, \
             patch.object(self.get_heatmaps, 'get_model_data') as mock_get_model_data, \
             patch.object(self.get_heatmaps.ImageDataBunch, 'from_folder') as mock_from_folder, \
             patch.object(self.get_heatmaps, 'hooked_backward') as mock_hooked_backward, \
             patch.object(self.get_heatmaps, 'save_img') as mock_save_img, \
             patch.object(self.get_heatmaps, 'show_heatmap') as mock_show_heatmap, \
             patch.object(self.get_heatmaps, 'plt') as mock_plt:

            mock_plt.subplots.return_value = (MagicMock(), MagicMock())

            mock_path_mod.exists.return_value = False
            mock_listdir.return_value = ['img1.jpg', 'img2.png']

            mock_learn = MagicMock()
            mock_learn.to_fp32.return_value = mock_learn
            mock_learn.model.eval.return_value = MagicMock()
            mock_data = MagicMock()
            mock_get_model_data.return_value = (mock_learn, mock_data)

            mock_temp = MagicMock()
            mock_temp.train_ds = [('x1', 'y1'), ('x2', 'y2')]
            mock_temp.one_item.return_value = [MagicMock()]
            mock_temp.denorm.return_value = [MagicMock()]

            mock_from_folder_result = MagicMock()
            mock_from_folder_result.normalize.return_value = mock_temp
            mock_from_folder.return_value = mock_from_folder_result

            mock_hook_a = MagicMock()
            mock_hook_a.stored = [MagicMock()]
            mock_hook_a.stored[0].cpu.return_value.mean.return_value = 'avg_acts'
            mock_hooked_backward.return_value = (mock_hook_a, MagicMock())

            with patch('argparse.ArgumentParser.parse_args') as mock_args:
                mock_args.return_value = MagicMock(
                    path_base='/base',
                    path_img='/img',
                    path_hms='/hms',
                    alpha=0.8
                )
                self.get_heatmaps.main()

            mock_get_model_data.assert_called_once()
            self.assertEqual(mock_save_img.call_count, 2)
            self.assertEqual(mock_show_heatmap.call_count, 2)
            mock_rmtree.assert_called_once()
            self.assertTrue(mock_mkdir.called)
            self.assertTrue(mock_rename.called)

if __name__ == '__main__':
    unittest.main()
