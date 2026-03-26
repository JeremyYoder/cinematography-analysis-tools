import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib

class TestGetPreds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock heavy dependencies before importing get-preds
        cls.mock_fastai = MagicMock()
        cls.mock_fastai_vision = MagicMock()
        cls.mock_pandas = MagicMock()

        sys.modules['fastai'] = cls.mock_fastai
        sys.modules['fastai.vision'] = cls.mock_fastai_vision
        sys.modules['pandas'] = cls.mock_pandas

        # Also mock initialise module dependencies if needed, although
        # importlib will re-evaluate get-preds imports
        sys.modules['initialise'] = MagicMock()

        # Import the script module dynamically
        cls.get_preds = importlib.import_module('get-preds')

    @classmethod
    def tearDownClass(cls):
        # Restore sys.modules
        del sys.modules['fastai']
        del sys.modules['fastai.vision']
        del sys.modules['pandas']
        del sys.modules['initialise']

    @patch('os.listdir')
    @patch('os.chdir')
    @patch('os.mkdir')
    @patch('os.path.exists')
    def test_save_preds_concat(self, mock_exists, mock_mkdir, mock_chdir, mock_listdir):
        # Setup mocks
        mock_exists.return_value = True # Pretend directory exists so no mkdir
        mock_listdir.return_value = ['img1.jpg', 'img2.jpg']

        # Mock learn and data objects
        mock_learn = MagicMock()
        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        # mock learn.predict(x)[2].numpy()
        mock_pred_tensor = MagicMock()
        mock_pred_tensor.numpy.return_value = [0.1, 0.2, 0.3, 0.4, 0.0]
        mock_learn.predict.return_value = [None, None, mock_pred_tensor]

        # Setup pandas mocks
        mock_df_instance = MagicMock()
        mock_df_instance.sort_values.return_value = mock_df_instance
        mock_df_instance.reset_index.return_value = mock_df_instance
        mock_df_instance.head.return_value = mock_df_instance
        self.mock_pandas.DataFrame.return_value = mock_df_instance

        mock_concat_df = MagicMock()
        self.mock_pandas.concat.return_value = mock_concat_df

        # Call the function
        path_img = '/mock/path/img'
        path_preds = '/mock/path/preds'

        # We need to mock open_image which is imported in get-preds
        with patch.object(self.get_preds, 'open_image', create=True) as mock_open_image:
            self.get_preds.save_preds(path_img, mock_learn, mock_data, path_preds=path_preds)

            # Assertions
            mock_chdir.assert_called_once_with(path_img)
            self.assertEqual(mock_open_image.call_count, 2)
            self.assertEqual(mock_learn.predict.call_count, 2)

            # Verify pandas concat was called
            self.mock_pandas.concat.assert_called_once()

            # Verify to_csv was called on the concatenated dataframe
            mock_concat_df.to_csv.assert_called_once()

if __name__ == '__main__':
    unittest.main()
