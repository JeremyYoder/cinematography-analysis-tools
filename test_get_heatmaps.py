import sys
import unittest
from unittest.mock import MagicMock, patch
import importlib

class TestGetHeatmaps(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_fastai = MagicMock()
        cls.mock_fastai_vision = MagicMock()
        cls.mock_torch = MagicMock()
        cls.mock_pandas = MagicMock()
        cls.mock_matplotlib = MagicMock()
        cls.mock_matplotlib_pyplot = MagicMock()
        cls.mock_matplotlib_ticker = MagicMock()
        cls.mock_fastai_callbacks = MagicMock()
        cls.mock_fastai_callbacks_hooks = MagicMock()

        sys.modules['fastai'] = cls.mock_fastai
        sys.modules['fastai.vision'] = cls.mock_fastai_vision
        sys.modules['fastai.callbacks'] = cls.mock_fastai_callbacks
        sys.modules['fastai.callbacks.hooks'] = cls.mock_fastai_callbacks_hooks
        sys.modules['torch'] = cls.mock_torch
        sys.modules['pandas'] = cls.mock_pandas
        sys.modules['matplotlib'] = cls.mock_matplotlib
        sys.modules['matplotlib.pyplot'] = cls.mock_matplotlib_pyplot
        sys.modules['matplotlib.ticker'] = cls.mock_matplotlib_ticker

        # Need to mock initialise so get_heatmaps.py can import from it
        cls.mock_initialise = MagicMock()
        sys.modules['initialise'] = cls.mock_initialise

        # Now we can import the module
        cls.get_heatmaps = importlib.import_module('get-heatmaps')

    @classmethod
    def tearDownClass(cls):
        del sys.modules['fastai']
        del sys.modules['fastai.vision']
        del sys.modules['fastai.callbacks']
        del sys.modules['fastai.callbacks.hooks']
        del sys.modules['torch']
        del sys.modules['pandas']
        del sys.modules['matplotlib']
        del sys.modules['matplotlib.pyplot']
        del sys.modules['matplotlib.ticker']
        del sys.modules['initialise']


    def test_hooked_backward(self):
        # mock hook_output using a context manager mock
        hook_output_cm = MagicMock()
        hook_output_instance = MagicMock()
        hook_output_cm.__enter__.return_value = hook_output_instance
        self.mock_fastai_callbacks_hooks.hook_output.return_value = hook_output_cm

        self.get_heatmaps.hook_output = self.mock_fastai_callbacks_hooks.hook_output

        # Setup mock inputs
        mock_m = MagicMock()
        mock_xb = MagicMock()
        cat = 2

        # Setup the predictions and backward call
        mock_preds = MagicMock()
        mock_m.return_value = mock_preds
        mock_backward_target = MagicMock()
        # mock_preds[0, 2] returns mock_backward_target
        mock_preds.__getitem__.return_value = mock_backward_target

        # Call the function
        hook_a, hook_g = self.get_heatmaps.hooked_backward(mock_m, mock_xb, cat)

        # Assertions
        self.assertEqual(hook_a, hook_output_instance)
        self.assertEqual(hook_g, hook_output_instance)

        # Verify hook_output was called twice (once for activations, once for gradients)
        self.assertEqual(self.mock_fastai_callbacks_hooks.hook_output.call_count, 2)

        # Verify model was called with xb
        mock_m.assert_called_with(mock_xb)

        # Verify backward was called on the correct element
        # We check that preds[0, 2].backward() was called
        mock_preds.__getitem__.assert_called_with((0, 2))
        mock_backward_target.backward.assert_called_once()

if __name__ == '__main__':
    unittest.main()
