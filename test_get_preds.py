import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib

import os
from pathlib import Path

class TestGetPreds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock pandas
        cls.mock_pd = MagicMock()
        cls.orig_pandas = sys.modules.get('pandas')
        sys.modules['pandas'] = cls.mock_pd

        # Mock fastai and fastai.vision
        cls.mock_fastai = MagicMock()
        cls.mock_fastai_vision = MagicMock()
        cls.orig_fastai = sys.modules.get('fastai')
        cls.orig_fastai_vision = sys.modules.get('fastai.vision')
        sys.modules['fastai'] = cls.mock_fastai
        sys.modules['fastai.vision'] = cls.mock_fastai_vision

        # Set up mock open_image and other components
        cls.mock_open_image = MagicMock()
        cls.mock_fastai_vision.open_image = cls.mock_open_image

        # Mock initialise
        cls.mock_initialise = MagicMock()
        cls.orig_initialise = sys.modules.get('initialise')
        sys.modules['initialise'] = cls.mock_initialise

        # Need to reload get-preds using importlib because of hyphen
        # Now that mocks are in place, we can safely import
        global get_preds
        get_preds = importlib.import_module("get-preds")

    @classmethod
    def tearDownClass(cls):
        # Restore original modules
        if cls.orig_pandas:
            sys.modules['pandas'] = cls.orig_pandas
        else:
            del sys.modules['pandas']

        if cls.orig_fastai:
            sys.modules['fastai'] = cls.orig_fastai
        else:
            del sys.modules['fastai']

        if cls.orig_fastai_vision:
            sys.modules['fastai.vision'] = cls.orig_fastai_vision
        else:
            del sys.modules['fastai.vision']

        if cls.orig_initialise:
            sys.modules['initialise'] = cls.orig_initialise
        else:
            del sys.modules['initialise']
    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.mkdir')
    def test_save_preds_with_path_preds(self, mock_mkdir, mock_exists, mock_listdir):
        mock_listdir.return_value = ['image1.jpg', 'image2.png', 'not_an_image.txt']
        mock_exists.return_value = False

        mock_learn = MagicMock()
        mock_learn.predict.return_value = (None, None, MagicMock(numpy=lambda: [0.1, 0.2, 0.7, 0.0, 0.0]))

        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        # Mock pandas DataFrame and concat
        mock_df = MagicMock()
        mock_df.sort_values.return_value = mock_df
        mock_df.reset_index.return_value = mock_df
        mock_df.head.return_value = mock_df

        # The mock_pd was used by get_preds, need to patch the methods
        with patch.object(get_preds.pd, 'DataFrame', return_value=mock_df), \
             patch.object(get_preds.pd, 'Categorical'), \
             patch.object(get_preds.pd, 'concat') as mock_concat:

            mock_concat.return_value = mock_df

            get_preds.save_preds(mock_learn, mock_data, '/fake/img/dir', '/fake/preds/dir')

            # Check if predictions directory was created
            mock_mkdir.assert_called_once_with('/fake/preds/dir')

            # Check if open_image was called for the two image files
            self.assertEqual(get_preds.open_image.call_count, 2)

            # Check if predictions were made for both images
            self.assertEqual(mock_learn.predict.call_count, 2)

            # Check if to_csv was called on the final dataframe with correct path
            mock_df.to_csv.assert_called_once()
            args, kwargs = mock_df.to_csv.call_args
            self.assertEqual(args[0], Path('/fake/preds/dir') / "preds.csv")
            self.assertEqual(kwargs['index'], False)

    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('os.mkdir')
    def test_save_preds_without_path_preds(self, mock_mkdir, mock_exists, mock_listdir):
        mock_listdir.return_value = ['image1.jpg']

        mock_learn = MagicMock()
        mock_learn.predict.return_value = (None, None, MagicMock(numpy=lambda: [0.1, 0.2, 0.7, 0.0, 0.0]))

        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        mock_df = MagicMock()
        mock_df.sort_values.return_value = mock_df
        mock_df.reset_index.return_value = mock_df
        mock_df.head.return_value = mock_df

        with patch.object(get_preds.pd, 'DataFrame', return_value=mock_df), \
             patch.object(get_preds.pd, 'Categorical'), \
             patch.object(get_preds.pd, 'concat') as mock_concat:

            mock_concat.return_value = mock_df

            get_preds.save_preds(mock_learn, mock_data, '/fake/img/dir')

            # mkdir should not be called if path_preds is None
            mock_mkdir.assert_not_called()

            # Check if to_csv was called on the final dataframe with correct path (path_img)
            mock_df.to_csv.assert_called_once()
            args, kwargs = mock_df.to_csv.call_args
            self.assertEqual(args[0], Path('/fake/img/dir') / "preds.csv")

    @patch('os.listdir')
    def test_save_preds_empty_directory(self, mock_listdir):
        mock_listdir.return_value = []

        mock_learn = MagicMock()
        mock_data = MagicMock()

        mock_empty_df = MagicMock()

        with patch.object(get_preds.pd, 'DataFrame', return_value=mock_empty_df), \
             patch.object(get_preds.pd, 'concat') as mock_concat:

            get_preds.save_preds(mock_learn, mock_data, '/fake/img/dir')

            # concat should not be called if there are no dataframes to append
            mock_concat.assert_not_called()

            # to_csv should be called on the empty dataframe
            mock_empty_df.to_csv.assert_called_once()

if __name__ == '__main__':
    unittest.main()