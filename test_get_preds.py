import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib

class TestGetPreds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock heavy dependencies globally for this test suite
        cls.mock_fastai = MagicMock()
        cls.mock_fastai_vision = MagicMock()
        cls.mock_pandas = MagicMock()
        cls.mock_torch = MagicMock()

        # Save original modules to restore later if needed
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

        from functools import partial
        cls.mock_fastai_vision.partial = partial

        # Mocking specific functions and classes in fastai.vision that initialise.py expects
        for func in ['cutout', 'jitter', 'skew', 'squish', 'tilt',
                     'perspective_warp', 'crop_pad', 'rgb_randomize',
                     'get_transforms', 'ImageDataBunch', 'ResizeMethod',
                     'imagenet_stats', 'cnn_learner', 'models', 'accuracy', 'open_image', 'Path']:
            setattr(cls.mock_fastai_vision, func, MagicMock(name=func))

        # Import the script
        cls.get_preds = importlib.import_module('get-preds')

    @classmethod
    def tearDownClass(cls):
        # Restore sys.modules
        for mod, original in cls.original_modules.items():
            if original is None:
                del sys.modules[mod]
            else:
                sys.modules[mod] = original

    @patch('os.listdir')
    @patch('os.chdir')
    @patch('os.path.exists')
    @patch('os.mkdir')
    def test_save_preds_creates_directory_if_preds_path_given(self, mock_mkdir, mock_exists, mock_chdir, mock_listdir):
        # Setup mocks
        mock_listdir.return_value = [] # No files to process
        mock_exists.return_value = False # Directory doesn't exist

        mock_learn = MagicMock()
        mock_data = MagicMock()

        # Call the function
        self.get_preds.save_preds(mock_learn, mock_data, '/fake/img/path', '/fake/preds/path')

        # Verify directory creation
        mock_exists.assert_called_with('/fake/preds/path')
        mock_mkdir.assert_called_with('/fake/preds/path')
        mock_chdir.assert_called_with('/fake/img/path')

    @patch('os.listdir')
    @patch('os.chdir')
    def test_save_preds_processes_images(self, mock_chdir, mock_listdir):
        # Setup mocks
        mock_listdir.return_value = ['image1.jpg', 'image2.png', 'not_an_image.txt']

        mock_learn = MagicMock()
        # Mock the predict method to return a tuple where the 3rd element has a .numpy() method
        mock_tensor = MagicMock()
        mock_tensor.numpy.return_value = [0.1, 0.2, 0.5, 0.1, 0.1]
        mock_learn.predict.return_value = (None, None, mock_tensor)

        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        # Mock pandas dataframe and concatenation
        mock_df = MagicMock()
        mock_df.sort_values.return_value = mock_df
        mock_df.reset_index.return_value = mock_df
        mock_df.head.return_value = mock_df

        self.mock_pandas.DataFrame.return_value = mock_df
        self.mock_pandas.concat.return_value = mock_df
        self.mock_pandas.Categorical = MagicMock()

        # Mock open_image from fastai since initialise.py does `from fastai.vision import *`
        # Because get-preds.py does `from initialise import *`, open_image is available globally in get-preds
        # But we need to patch it in get_preds namespace or provide it
        self.get_preds.open_image = MagicMock()

        # Call the function
        self.get_preds.save_preds(mock_learn, mock_data, '/fake/img/path')

        # Verify open_image was called for each image
        self.assertEqual(self.get_preds.open_image.call_count, 2)

        # Verify pandas operations
        self.assertTrue(self.mock_pandas.DataFrame.called)
        self.assertTrue(self.mock_pandas.concat.called)

        # Verify output saved to img path since preds path is None
        self.assertTrue(mock_df.to_csv.called)

        # We need to extract the path argument from to_csv call to verify it correctly
        # because Path objects are used which we mocked
        to_csv_args, to_csv_kwargs = mock_df.to_csv.call_args
        self.assertEqual(to_csv_kwargs.get('index'), False)

if __name__ == '__main__':
    unittest.main()
