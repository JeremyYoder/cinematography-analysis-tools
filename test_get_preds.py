import os
import sys
import unittest
import importlib
from unittest.mock import MagicMock, patch
from pathlib import Path

# Create a mock objects for heavy dependencies
mock_fastai = MagicMock()
mock_fastai_vision = MagicMock()
mock_initialise = MagicMock()
mock_pandas = MagicMock()

class TestGetPreds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock sys.modules
        sys.modules['fastai'] = mock_fastai
        sys.modules['fastai.vision'] = mock_fastai_vision
        sys.modules['initialise'] = mock_initialise
        sys.modules['pandas'] = mock_pandas

        # Import get-preds module
        cls.get_preds = importlib.import_module('get-preds')

    @classmethod
    def tearDownClass(cls):
        # Restore sys.modules
        del sys.modules['fastai']
        del sys.modules['fastai.vision']
        del sys.modules['initialise']
        del sys.modules['pandas']

    def setUp(self):
        # Reset mock calls between tests
        mock_fastai_vision.reset_mock()
        mock_initialise.reset_mock()

    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('pandas.DataFrame.to_csv')
    @patch('pandas.concat')
    @patch('os.getcwd')
    @patch('os.chdir')
    def test_save_preds_with_path_preds(self, mock_chdir, mock_getcwd, mock_concat, mock_to_csv, mock_mkdir, mock_exists, mock_listdir):
        # Setup mocks
        mock_getcwd.return_value = '/mock/cwd'
        mock_listdir.return_value = ['image1.jpg', 'image2.png']
        mock_exists.return_value = False

        mock_learn = MagicMock()
        mock_learn.predict.return_value = [None, None, MagicMock(numpy=lambda: [0.1, 0.2, 0.7, 0.0, 0.0])]

        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        # We need to mock pandas concat to return a proper dataframe
        mock_bdf = MagicMock()
        mock_concat.return_value = mock_bdf

        # Call function
        path_img = '/mock/img/path'
        path_preds = '/mock/preds/path'
        self.get_preds.save_preds(mock_learn, mock_data, path_img, path_preds)

        # Verify mkdir was called since path_preds does not exist
        mock_mkdir.assert_called_once_with(path_preds)

        # Verify chdir calls
        mock_chdir.assert_any_call(path_img)
        mock_chdir.assert_any_call('/mock/cwd')

        # Verify DataFrame generation and saving
        mock_concat.assert_called_once()
        mock_bdf.to_csv.assert_called_once_with(Path(path_preds) / 'preds.csv', index=False)

    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('os.getcwd')
    @patch('os.chdir')
    def test_save_preds_empty_dir(self, mock_chdir, mock_getcwd, mock_mkdir, mock_exists, mock_listdir):
        # Setup mocks
        mock_getcwd.return_value = '/mock/cwd'
        mock_listdir.return_value = ['not_an_image.txt']
        mock_exists.return_value = True

        mock_learn = MagicMock()
        mock_data = MagicMock()

        # We need to mock pandas DataFrame creation
        mock_bdf = MagicMock()
        mock_pandas.DataFrame.return_value = mock_bdf

        # Call function
        path_img = '/mock/img/path'
        path_preds = '/mock/preds/path'
        self.get_preds.save_preds(mock_learn, mock_data, path_img, path_preds)

        # Verify mkdir was NOT called since path_preds exists
        mock_mkdir.assert_not_called()

        # Verify chdir calls
        mock_chdir.assert_any_call(path_img)
        mock_chdir.assert_any_call('/mock/cwd')

        # Verify to_csv called with empty DataFrame
        mock_bdf.to_csv.assert_called_once_with(Path(path_preds) / 'preds.csv', index=False)


if __name__ == '__main__':
    unittest.main()
