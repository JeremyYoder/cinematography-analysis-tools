import sys
import unittest
from unittest.mock import MagicMock, patch
import os
import importlib

class TestGetPreds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_fastai = MagicMock()
        cls.mock_fastai_vision = MagicMock()
        cls.mock_pandas = MagicMock()
        cls.mock_torch = MagicMock()

        cls.original_modules = {
            'fastai': sys.modules.get('fastai'),
            'fastai.vision': sys.modules.get('fastai.vision'),
            'pandas': sys.modules.get('pandas'),
            'torch': sys.modules.get('torch'),
        }

        sys.modules['fastai'] = cls.mock_fastai
        sys.modules['fastai.vision'] = cls.mock_fastai_vision
        sys.modules['pandas'] = cls.mock_pandas
        sys.modules['torch'] = cls.mock_torch

        global get_preds
        get_preds = importlib.import_module('get-preds')

    @classmethod
    def tearDownClass(cls):
        for name, module in cls.original_modules.items():
            if module is None:
                del sys.modules[name]
            else:
                sys.modules[name] = module

    def setUp(self):
        self.mock_pandas.reset_mock()
        self.mock_fastai_vision.reset_mock()

    @patch('os.listdir')
    @patch('os.chdir')
    @patch('os.getcwd')
    @patch('os.path.exists')
    @patch('os.mkdir')
    def test_save_preds(self, mock_mkdir, mock_exists, mock_getcwd, mock_chdir, mock_listdir):
        # Setup mocks
        mock_getcwd.return_value = '/mock/orig/dir'
        mock_listdir.return_value = ['img1.jpg', 'img2.jpg', 'not_img.txt']
        mock_exists.return_value = False

        mock_learn = MagicMock()
        mock_predict = MagicMock()
        mock_predict.return_value = [None, None, MagicMock(numpy=MagicMock(return_value=MagicMock()))]
        mock_learn.predict = mock_predict

        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        # DataFrame mock behavior
        mock_df_instance = MagicMock()
        self.mock_pandas.DataFrame.return_value = mock_df_instance
        mock_df_instance.__getitem__.return_value = mock_df_instance
        mock_df_instance.idxmax.return_value = ['LS', 'FS']
        mock_df_instance.max.return_value = [90, 80]

        # Call the function
        get_preds.save_preds('/mock/img/dir', '/mock/preds/dir', mock_learn, mock_data)

        # Assertions
        mock_mkdir.assert_called_with('/mock/preds/dir')
        mock_chdir.assert_any_call('/mock/img/dir')
        mock_chdir.assert_any_call('/mock/orig/dir')

        # Since os.listdir filters to .jpg, .jpeg, .png, we have 2 images
        self.assertEqual(mock_learn.predict.call_count, 2)
        self.mock_pandas.DataFrame.assert_called()
        mock_df_instance.to_csv.assert_called_once()
        args, kwargs = mock_df_instance.to_csv.call_args
        self.assertTrue(str(args[0]).endswith('preds.csv'))
        self.assertEqual(kwargs['index'], False)

if __name__ == '__main__':
    unittest.main()
