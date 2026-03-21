import sys
import unittest
from unittest.mock import MagicMock, patch
from functools import partial

class TestInitialise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock fastai.vision and its components before importing initialise
        cls.mock_fastai = MagicMock()
        cls.mock_fastai_vision = MagicMock()

        cls.patcher = patch.dict('sys.modules', {
            'fastai': cls.mock_fastai,
            'fastai.vision': cls.mock_fastai_vision
        })
        cls.patcher.start()

        # Define mocks for functions used in initialise.py
        mock_functions = [
            'cutout', 'jitter', 'skew', 'squish', 'tilt',
            'perspective_warp', 'crop_pad', 'rgb_randomize',
            'get_transforms', 'ImageDataBunch', 'ResizeMethod',
            'imagenet_stats', 'cnn_learner', 'models', 'accuracy'
        ]

        for func in mock_functions:
            setattr(cls.mock_fastai_vision, func, MagicMock(name=func))

        # fastai.vision typically exports partial from functools
        cls.mock_fastai_vision.partial = partial

        # Import initialise after mocking
        global initialise
        import initialise
        # Inject fastai.vision * into initialise namespace
        for func in mock_functions:
            setattr(initialise, func, getattr(cls.mock_fastai_vision, func))
        initialise.partial = partial
        cls.initialise = initialise

    @classmethod
    def tearDownClass(cls):
        cls.patcher.stop()
        if 'initialise' in sys.modules:
            del sys.modules['initialise']

    def test_xtra_tfms_default(self):
        tfms = self.initialise.xtra_tfms()
        self.assertEqual(len(tfms), 7)
        # Verify that the transforms are called (they are partials, and then called)
        # In xtra_tfms, they are defined as partials and then called: jitter_(), etc.
        # Since we mocked the base functions as MagicMocks, calling the partial returns a mock call.

    def test_xtra_tfms_base_size(self):
        base_size = 400
        expected_box_dim = int(base_size / 4) # 100

        with patch('initialise.partial', wraps=partial) as mock_partial:
            tfms = self.initialise.xtra_tfms(base_size=base_size)

            # Check if cutout was called with the correct length
            # cutout_ = partial(cutout, p = .8, n_holes = (1,1), length = (box_dim, box_dim))
            # Find the call to partial that uses cutout
            cutout_call = None
            for call in mock_partial.call_args_list:
                if call.args[0] == self.mock_fastai_vision.cutout:
                    cutout_call = call
                    break

            self.assertIsNotNone(cutout_call, "cutout was not used in a partial")
            self.assertEqual(cutout_call.kwargs['length'], (expected_box_dim, expected_box_dim))

    def test_xtra_tfms_returns_list(self):
        tfms = self.initialise.xtra_tfms()
        self.assertIsInstance(tfms, list)
        for tfm in tfms:
            # Since they are results of calling a MagicMock (the partial),
            # they should be MagicMock objects
            self.assertIsInstance(tfm, MagicMock)

if __name__ == '__main__':
    unittest.main()
