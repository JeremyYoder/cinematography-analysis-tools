import sys
import unittest
from unittest.mock import MagicMock, patch
import os
import pandas as pd
from pathlib import Path
import importlib

# Mock heavy dependencies BEFORE importing the module
mock_fastai = MagicMock()
mock_fastai_vision = MagicMock()
sys.modules['fastai'] = mock_fastai
sys.modules['fastai.vision'] = mock_fastai_vision

mock_pandas = MagicMock()
sys.modules['pandas'] = mock_pandas

mock_initialise = MagicMock()
sys.modules['initialise'] = mock_initialise

# Now we can import the module safely
get_preds = importlib.import_module('get-preds')


class TestGetPreds(unittest.TestCase):

    @patch('os.listdir')
    @patch('os.mkdir')
    @patch('os.chdir')
    @patch('os.path.exists')
    @patch.object(get_preds, 'open_image')
    @patch.object(get_preds, 'pd')
    def test_save_preds(self, mock_pd, mock_open_image, mock_exists, mock_chdir, mock_mkdir, mock_listdir):
        # Setup mocks
        mock_exists.return_value = False
        mock_listdir.return_value = ['test1.jpg', 'test2.png']

        mock_learn = MagicMock()
        mock_learn.predict.return_value = [None, None, MagicMock(numpy=lambda: [0.1, 0.8, 0.05, 0.03, 0.02])]

        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        # Setup pandas dataframe mock chain to simulate what happens in the code
        mock_df = MagicMock()
        mock_df.sort_values.return_value.reset_index.return_value = mock_df
        mock_df.sort_values.return_value = mock_df
        mock_df.head.return_value = mock_df

        mock_pd.DataFrame.return_value = mock_df
        mock_pd.Categorical.return_value = 'categorical'

        mock_concat_df = MagicMock()
        mock_pd.concat.return_value = mock_concat_df

        # Run the function
        path_img = '/test/img/path'
        path_preds = '/test/preds/path'

        get_preds.save_preds(path_img, mock_learn, mock_data, path_preds)

        # Assertions
        mock_mkdir.assert_called_once_with(path_preds)
        mock_chdir.assert_called_once_with(path_img)

        self.assertEqual(mock_open_image.call_count, 2)
        mock_open_image.assert_any_call('test1.jpg')
        mock_open_image.assert_any_call('test2.png')

        self.assertEqual(mock_learn.predict.call_count, 2)

        self.assertEqual(mock_pd.DataFrame.call_count, 2)

        # Verify the final dataframe was saved
        mock_pd.concat.assert_called_once()
        mock_concat_df.to_csv.assert_called_once_with(Path(path_preds) / 'preds.csv', index=False)

    @patch('os.listdir')
    @patch('os.mkdir')
    @patch('os.chdir')
    @patch('os.path.exists')
    @patch.object(get_preds, 'open_image')
    @patch.object(get_preds, 'pd')
    def test_save_preds_no_path_preds(self, mock_pd, mock_open_image, mock_exists, mock_chdir, mock_mkdir, mock_listdir):
        # Setup mocks
        mock_listdir.return_value = ['test1.jpg']

        mock_learn = MagicMock()
        mock_learn.predict.return_value = [None, None, MagicMock(numpy=lambda: [0.1, 0.8, 0.05, 0.03, 0.02])]

        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        mock_df = MagicMock()
        mock_df.sort_values.return_value.reset_index.return_value = mock_df
        mock_df.sort_values.return_value = mock_df
        mock_df.head.return_value = mock_df

        mock_pd.DataFrame.return_value = mock_df
        mock_pd.Categorical.return_value = 'categorical'

        mock_concat_df = MagicMock()
        mock_pd.concat.return_value = mock_concat_df

        # Run the function
        path_img = '/test/img/path'

        get_preds.save_preds(path_img, mock_learn, mock_data)

        # Assertions
        mock_mkdir.assert_not_called()
        mock_chdir.assert_called_once_with(path_img)

        mock_concat_df.to_csv.assert_called_once_with(Path(path_img) / 'preds.csv', index=False)

if __name__ == '__main__':
    unittest.main()
