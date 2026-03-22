import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib

class TestGetPreds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_fastai = MagicMock()
        cls.mock_fastai_vision = MagicMock()
        cls.mock_torch = MagicMock()
        cls.mock_pandas = MagicMock()

        sys.modules['fastai'] = cls.mock_fastai
        sys.modules['fastai.vision'] = cls.mock_fastai_vision
        sys.modules['torch'] = cls.mock_torch

        # mock pandas dataframe and concat
        cls.mock_pandas.DataFrame = MagicMock()
        cls.mock_pandas.Categorical = MagicMock()
        cls.mock_pandas.concat = MagicMock()
        sys.modules['pandas'] = cls.mock_pandas

        # Need to mock initialise so get_preds.py can import from it
        cls.mock_initialise = MagicMock()
        cls.mock_initialise.open_image = MagicMock()
        cls.mock_initialise.Path = MagicMock()
        sys.modules['initialise'] = cls.mock_initialise

        # Now we can import the module
        cls.get_preds = importlib.import_module('get-preds')

    @classmethod
    def tearDownClass(cls):
        del sys.modules['fastai']
        del sys.modules['fastai.vision']
        del sys.modules['torch']
        del sys.modules['pandas']
        del sys.modules['initialise']


    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('os.chdir')
    def test_save_preds(self, mock_chdir, mock_mkdir, mock_exists, mock_listdir):
        # We need to mock open_image globally for test
        self.mock_initialise.open_image.reset_mock()
        self.get_preds.open_image = self.mock_initialise.open_image
        self.get_preds.Path = self.mock_initialise.Path

        # Setup mocks
        mock_listdir.return_value = ['image1.jpg', 'image2.png']
        mock_exists.return_value = False # to trigger mkdir if path_preds is provided

        mock_learn = MagicMock()
        mock_learn.predict.return_value = [None, None, MagicMock()]
        mock_learn.predict()[2].numpy.return_value = [0.1, 0.2, 0.7, 0.0, 0.0]

        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        # Setup pandas dataframe mock
        mock_df_instance = MagicMock()
        self.mock_pandas.DataFrame.return_value = mock_df_instance
        mock_df_instance.sort_values.return_value = mock_df_instance
        mock_df_instance.reset_index.return_value = mock_df_instance
        mock_df_instance.head.return_value = mock_df_instance

        mock_bdf_instance = MagicMock()
        self.mock_pandas.concat.return_value = mock_bdf_instance

        # Execute
        self.get_preds.save_preds(mock_learn, mock_data, '/fake/img/path', '/fake/preds/path')

        # Assertions
        mock_mkdir.assert_called_with('/fake/preds/path')
        mock_chdir.assert_called_with('/fake/img/path')
        self.assertEqual(self.mock_initialise.open_image.call_count, 2)
        # 2 from the loop, plus 1 from setting up the mock
        self.assertEqual(mock_learn.predict.call_count, 3)
        mock_bdf_instance.to_csv.assert_called_once()

    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('os.chdir')
    def test_save_preds_no_path_preds(self, mock_chdir, mock_mkdir, mock_exists, mock_listdir):
        self.mock_initialise.open_image.reset_mock()
        self.get_preds.open_image = self.mock_initialise.open_image
        self.get_preds.Path = self.mock_initialise.Path

        # Setup mocks
        mock_listdir.return_value = ['image1.jpg', 'image2.png']
        mock_exists.return_value = True

        mock_learn = MagicMock()
        mock_learn.predict.return_value = [None, None, MagicMock()]
        mock_learn.predict()[2].numpy.return_value = [0.1, 0.2, 0.7, 0.0, 0.0]

        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        # Setup pandas dataframe mock
        mock_df_instance = MagicMock()
        self.mock_pandas.DataFrame.return_value = mock_df_instance
        mock_df_instance.sort_values.return_value = mock_df_instance
        mock_df_instance.reset_index.return_value = mock_df_instance
        mock_df_instance.head.return_value = mock_df_instance

        mock_bdf_instance = MagicMock()
        self.mock_pandas.concat.return_value = mock_bdf_instance

        # Execute
        self.get_preds.save_preds(mock_learn, mock_data, '/fake/img/path', None)

        # Assertions
        mock_mkdir.assert_not_called()
        mock_chdir.assert_called_with('/fake/img/path')
        self.assertEqual(self.mock_initialise.open_image.call_count, 2)
        # 2 from the loop, plus 1 from setting up the mock
        self.assertEqual(mock_learn.predict.call_count, 3)
        mock_bdf_instance.to_csv.assert_called_once()

if __name__ == '__main__':
    unittest.main()
