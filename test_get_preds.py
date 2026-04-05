import sys
import unittest
from unittest.mock import MagicMock, patch
import os
from pathlib import Path
from functools import partial

class TestGetPreds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock heavy dependencies
        cls.mock_fastai = MagicMock()
        cls.mock_fastai_vision = MagicMock()
        cls.mock_pandas = MagicMock()
        cls.mock_matplotlib = MagicMock()

        cls.original_modules = sys.modules.copy()

        sys.modules['fastai'] = cls.mock_fastai
        sys.modules['fastai.vision'] = cls.mock_fastai_vision
        sys.modules['fastai.callbacks.hooks'] = MagicMock()
        sys.modules['pandas'] = cls.mock_pandas
        sys.modules['matplotlib'] = cls.mock_matplotlib
        sys.modules['matplotlib.pyplot'] = MagicMock()
        sys.modules['matplotlib.ticker'] = MagicMock()

        mock_functions = [
            'cutout', 'jitter', 'skew', 'squish', 'tilt',
            'perspective_warp', 'crop_pad', 'rgb_randomize',
            'get_transforms', 'ImageDataBunch', 'ResizeMethod',
            'imagenet_stats', 'cnn_learner', 'models', 'accuracy',
            'Image', 'open_image'
        ]

        for func in mock_functions:
            setattr(cls.mock_fastai_vision, func, MagicMock(name=func))

        cls.mock_fastai_vision.partial = partial

        import importlib.util
        spec = importlib.util.spec_from_file_location("get_preds", "get-preds.py")
        cls.get_preds = importlib.util.module_from_spec(spec)
        sys.modules["get_preds"] = cls.get_preds
        spec.loader.exec_module(cls.get_preds)

    @classmethod
    def tearDownClass(cls):
        sys.modules.clear()
        sys.modules.update(cls.original_modules)

    def setUp(self):
        # Reset mocks before each test
        self.mock_pandas.DataFrame.reset_mock()
        self.mock_pandas.concat.reset_mock()
        self.mock_fastai_vision.open_image.reset_mock()
        self.mock_pandas.DataFrame.return_value.to_csv.reset_mock()
        self.mock_pandas.concat.return_value.to_csv.reset_mock()

    @patch('os.listdir')
    @patch('os.chdir')
    def test_save_preds(self, mock_chdir, mock_listdir):
        # Since pandas is globally mocked, we must patch the object methods via the mock object,
        # or verify calls on the global mock.
        # Setup mock dependencies
        mock_learn = MagicMock()
        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        # Mock predictions: probabilities for each class
        mock_preds = MagicMock()
        mock_preds.numpy.return_value = [0.1, 0.2, 0.5, 0.15, 0.05]
        mock_learn.predict.return_value = (None, None, mock_preds)

        mock_listdir.return_value = ['test1.jpg', 'test2.png', 'test3.txt']
        self.mock_fastai_vision.open_image.return_value = MagicMock()

        # Call save_preds
        path_img = '/fake/img/path'
        self.get_preds.save_preds(mock_learn, mock_data, path_img)

        # Verify chdir was called with path_img
        mock_chdir.assert_called_with(path_img)

        # Verify open_image was called for the image files only
        self.assertEqual(self.mock_fastai_vision.open_image.call_count, 2)
        self.mock_fastai_vision.open_image.assert_any_call('test1.jpg')
        self.mock_fastai_vision.open_image.assert_any_call('test2.png')

        # Verify predict was called
        self.assertEqual(mock_learn.predict.call_count, 2)

        # Verify to_csv was called once to save preds.csv on the concatenated dataframe
        self.assertEqual(self.mock_pandas.DataFrame.return_value.to_csv.call_count, 1)
        args, kwargs = self.mock_pandas.DataFrame.return_value.to_csv.call_args
        self.assertEqual(args[0], Path(path_img) / 'preds.csv')
        self.assertEqual(kwargs.get('index'), False)

    @patch('os.path.exists')
    @patch('os.mkdir')
    @patch('os.listdir')
    @patch('os.chdir')
    def test_save_preds_with_path_preds(self, mock_chdir, mock_listdir, mock_mkdir, mock_exists):
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

        self.get_preds.save_preds(mock_learn, mock_data, path_img, path_preds=path_preds)

        # Verify mkdir was called because exists returned False
        mock_mkdir.assert_called_once_with(path_preds)

        # Verify to_csv was called with path_preds on the concatenated dataframe
        self.assertEqual(self.mock_pandas.DataFrame.return_value.to_csv.call_count, 1)
        args, kwargs = self.mock_pandas.DataFrame.return_value.to_csv.call_args
        self.assertEqual(args[0], Path(path_preds) / 'preds.csv')

if __name__ == '__main__':
    unittest.main()
