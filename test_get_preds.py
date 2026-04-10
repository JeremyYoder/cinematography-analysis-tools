import sys
import unittest
from unittest.mock import MagicMock, patch
import os
from pathlib import Path

# Mock external dependencies globally before import
mock_pandas = MagicMock()
sys.modules['pandas'] = mock_pandas

mock_torch = MagicMock()
sys.modules['torch'] = mock_torch

# Mock fastai and its components
mock_fastai = MagicMock()
mock_fastai_vision = MagicMock()
sys.modules['fastai'] = mock_fastai
sys.modules['fastai.vision'] = mock_fastai_vision
sys.modules['fastai.callbacks'] = MagicMock()
sys.modules['fastai.callbacks.hooks'] = MagicMock()

# Define mocks for functions used in initialise.py
mock_functions = [
    'cutout', 'jitter', 'skew', 'squish', 'tilt',
    'perspective_warp', 'crop_pad', 'rgb_randomize',
    'get_transforms', 'ImageDataBunch', 'ResizeMethod',
    'imagenet_stats', 'cnn_learner', 'models', 'accuracy',
    'Image', 'open_image'
]

for func in mock_functions:
    setattr(mock_fastai_vision, func, MagicMock(name=func))

from functools import partial
mock_fastai_vision.partial = partial

import importlib.util

spec = importlib.util.spec_from_file_location("get_preds", "get-preds.py")
get_preds = importlib.util.module_from_spec(spec)
sys.modules["get_preds"] = get_preds
spec.loader.exec_module(get_preds)

class TestGetPreds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @patch('os.listdir')
    @patch('get_preds.open_image', create=True)
    @patch('os.chdir')
    def test_save_preds(self, mock_chdir, mock_open_image, mock_listdir):
        # Setup mock dependencies
        mock_learn = MagicMock()
        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        # Mock predictions: probabilities for each class
        mock_preds = MagicMock()
        mock_preds.numpy.return_value = [0.1, 0.2, 0.5, 0.15, 0.05]
        mock_learn.predict.return_value = (None, None, mock_preds)

        mock_listdir.return_value = ['test1.jpg', 'test2.png', 'test3.txt']
        mock_open_image.return_value = MagicMock()

        # Call save_preds
        path_img = '/fake/img/path'
        get_preds.save_preds(mock_learn, mock_data, path_img)

        # Verify chdir was called with Path object as per memory if applicable, wait get-preds calls os.chdir(path_img)
        # Verify open_image was called for the image files only
        self.assertEqual(mock_open_image.call_count, 2)
        mock_open_image.assert_any_call('test1.jpg')
        mock_open_image.assert_any_call('test2.png')

        # Verify predict was called
        self.assertEqual(mock_learn.predict.call_count, 2)

        # Verify to_csv was called once to save preds.csv
        # using the globally mocked pandas object
        mock_pandas.DataFrame.return_value.to_csv.assert_called_once()
        args, kwargs = mock_pandas.DataFrame.return_value.to_csv.call_args
        self.assertEqual(args[0], Path(path_img) / 'preds.csv')
        self.assertEqual(kwargs.get('index'), False)

    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('os.listdir')
    @patch('get_preds.open_image', create=True)
    @patch('os.chdir')
    def test_save_preds_with_path_preds(self, mock_chdir, mock_open_image, mock_listdir, mock_mkdir, mock_exists):
        mock_learn = MagicMock()
        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        mock_preds = MagicMock()
        mock_preds.numpy.return_value = [0.1, 0.2, 0.5, 0.15, 0.05]
        mock_learn.predict.return_value = (None, None, mock_preds)

        mock_exists.return_value = False
        mock_listdir.return_value = ['test1.jpg']

        path_img = '/fake/img/path'
        path_preds = '/fake/preds/path'

        get_preds.save_preds(mock_learn, mock_data, path_img, path_preds=path_preds)

        # Verify mkdir was called because exists returned False
        mock_mkdir.assert_called_once_with(path_preds)

        # Verify to_csv was called with path_preds
        mock_pandas.DataFrame.return_value.to_csv.assert_called()

if __name__ == '__main__':
    unittest.main()
