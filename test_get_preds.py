import sys
import unittest
from unittest.mock import MagicMock, patch
import os
import importlib.util

class TestGetPreds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We need to temporarily mock heavy dependencies just for importing get-preds.py
        # and then clean up to avoid polluting other tests
        cls.original_modules = dict(sys.modules)

        mock_fastai = MagicMock()
        mock_fastai_vision = MagicMock()
        mock_pandas = MagicMock()

        sys.modules['fastai'] = mock_fastai
        sys.modules['fastai.vision'] = mock_fastai_vision
        sys.modules['pandas'] = mock_pandas

        spec = importlib.util.spec_from_file_location("get_preds", "get-preds.py")
        cls.get_preds = importlib.util.module_from_spec(spec)
        sys.modules["get_preds"] = cls.get_preds

        cls.get_preds.pd = mock_pandas
        cls.get_preds.open_image = MagicMock()
        cls.get_preds.Path = MagicMock()

        spec.loader.exec_module(cls.get_preds)

    @classmethod
    def tearDownClass(cls):
        # Restore original sys.modules to prevent global state pollution
        to_remove = [k for k in sys.modules if k not in cls.original_modules]
        for k in to_remove:
            del sys.modules[k]
        sys.modules.update(cls.original_modules)


    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('os.chdir')
    def test_save_preds_with_path_preds(self, mock_chdir, mock_mkdir, mock_exists, mock_listdir):
        mock_exists.return_value = False
        mock_listdir.return_value = ['image1.jpg', 'image2.png', 'not_an_image.txt']

        mock_learn = MagicMock()
        mock_learn.predict.return_value = [None, None, MagicMock(numpy=MagicMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5]))]

        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        mock_df = MagicMock()
        mock_df.sort_values.return_value = mock_df
        mock_df.reset_index.return_value = mock_df
        mock_df.head.return_value = mock_df

        # Patch the pandas methods used in the module
        with patch.object(self.get_preds.pd, 'DataFrame', return_value=mock_df) as mock_dataframe, \
             patch.object(self.get_preds.pd, 'Categorical'), \
             patch.object(self.get_preds.pd, 'concat') as mock_concat, \
             patch.object(self.get_preds, 'open_image') as mock_open_image:

            mock_bdf = MagicMock()
            mock_concat.return_value = mock_bdf

            self.get_preds.save_preds('/test/img/path', mock_learn, mock_data, '/test/preds/path')

            mock_mkdir.assert_called_once_with('/test/preds/path')
            mock_chdir.assert_called_once_with('/test/img/path')
            self.assertEqual(mock_open_image.call_count, 2)
            mock_learn.predict.assert_called()
            mock_concat.assert_called_once()
            mock_bdf.to_csv.assert_called_once()

    @patch('os.listdir')
    @patch('os.chdir')
    def test_save_preds_without_path_preds(self, mock_chdir, mock_listdir):
        mock_listdir.return_value = ['image1.jpg']

        mock_learn = MagicMock()
        mock_learn.predict.return_value = [None, None, MagicMock(numpy=MagicMock(return_value=[0.1, 0.2, 0.3, 0.4, 0.5]))]

        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        mock_df = MagicMock()
        mock_df.sort_values.return_value = mock_df
        mock_df.reset_index.return_value = mock_df
        mock_df.head.return_value = mock_df

        with patch.object(self.get_preds.pd, 'DataFrame', return_value=mock_df) as mock_dataframe, \
             patch.object(self.get_preds.pd, 'Categorical'), \
             patch.object(self.get_preds.pd, 'concat') as mock_concat, \
             patch.object(self.get_preds, 'open_image') as mock_open_image, \
             patch.object(self.get_preds, 'Path') as mock_path:

            mock_bdf = MagicMock()
            mock_concat.return_value = mock_bdf

            self.get_preds.save_preds('/test/img/path', mock_learn, mock_data, None)

            mock_chdir.assert_called_once_with('/test/img/path')
            self.assertEqual(mock_open_image.call_count, 1)
            mock_learn.predict.assert_called()
            mock_concat.assert_called_once()
            mock_bdf.to_csv.assert_called_once()


if __name__ == '__main__':
    unittest.main()
