import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib

# Mock heavy dependencies globally
mock_pandas = MagicMock()
mock_torch = MagicMock()
mock_matplotlib = MagicMock()
mock_matplotlib_pyplot = MagicMock()
mock_matplotlib_ticker = MagicMock()
mock_fastai = MagicMock()
mock_fastai_vision = MagicMock()
mock_fastai_callbacks = MagicMock()
mock_fastai_callbacks_hooks = MagicMock()

sys.modules['pandas'] = mock_pandas
sys.modules['torch'] = mock_torch
sys.modules['matplotlib'] = mock_matplotlib
sys.modules['matplotlib.pyplot'] = mock_matplotlib_pyplot
sys.modules['matplotlib.ticker'] = mock_matplotlib_ticker
sys.modules['fastai'] = mock_fastai
sys.modules['fastai.vision'] = mock_fastai_vision
sys.modules['fastai.callbacks'] = mock_fastai_callbacks
sys.modules['fastai.callbacks.hooks'] = mock_fastai_callbacks_hooks

get_preds = importlib.import_module('get-preds')

class TestGetPreds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        # Reset mocks
        mock_pandas.reset_mock()
        mock_fastai_vision.reset_mock()

    @patch('os.listdir')
    @patch('os.makedirs')
    def test_save_preds_creates_directory(self, mock_makedirs, mock_listdir):
        mock_listdir.return_value = []
        get_preds.save_preds('img_dir', 'preds_dir', MagicMock(), MagicMock())
        mock_makedirs.assert_called_once_with('preds_dir', exist_ok=True)

    @patch('os.listdir')
    @patch('os.makedirs')
    def test_save_preds_processes_files(self, mock_makedirs, mock_listdir):
        mock_listdir.return_value = ['image1.jpg', 'image2.png']

        mock_learn = MagicMock()
        mock_predict = MagicMock()

        # Mock numpy return value
        class MockTensor:
            def numpy(self):
                return [0.1, 0.8, 0.05, 0.02, 0.03]

        mock_predict.return_value = [None, None, MockTensor()]
        mock_learn.predict = mock_predict

        mock_data = MagicMock()
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        mock_bdf = MagicMock()
        mock_pandas.DataFrame.return_value = mock_bdf

        with patch.object(get_preds, 'open_image') as mock_open_image:
            get_preds.save_preds('img_dir', None, mock_learn, mock_data)

            self.assertEqual(mock_open_image.call_count, 2)
            self.assertEqual(mock_learn.predict.call_count, 2)

            # DataFrame instantiation
            self.assertEqual(mock_pandas.DataFrame.call_count, 2)

            # Since no path_preds, writes to img_dir
            mock_pandas.DataFrame.return_value.to_csv.assert_called_once()
            args, kwargs = mock_pandas.DataFrame.return_value.to_csv.call_args
            self.assertEqual(str(args[0]).replace('\\', '/'), 'img_dir/preds.csv')
            self.assertEqual(kwargs['index'], False)

if __name__ == '__main__':
    unittest.main()
