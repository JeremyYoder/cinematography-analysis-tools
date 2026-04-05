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
        cls.original_modules = sys.modules.copy()

        sys.modules['fastai'] = cls.mock_fastai
        sys.modules['fastai.vision'] = cls.mock_fastai_vision

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

        import importlib.util
        spec = importlib.util.spec_from_file_location("initialise", "initialise.py")
        cls.initialise = importlib.util.module_from_spec(spec)
        sys.modules["initialise"] = cls.initialise
        spec.loader.exec_module(cls.initialise)

    @classmethod
    def tearDownClass(cls):
        sys.modules.clear()
        sys.modules.update(cls.original_modules)

    def test_xtra_tfms_default(self):
        tfms = self.initialise.xtra_tfms()
        self.assertEqual(len(tfms), 7)
        # Verify that the transforms are called (they are partials, and then called)
        # In xtra_tfms, they are defined as partials and then called: jitter_(), etc.
        # Since we mocked the base functions as MagicMocks, calling the partial returns a mock call.

    def test_xtra_tfms_base_size(self):
        base_size = 400
        expected_box_dim = int(base_size / 4) # 100

        self.mock_fastai_vision.cutout.reset_mock()
        tfms = self.initialise.xtra_tfms(base_size=base_size)

        self.mock_fastai_vision.cutout.assert_called_with(length=(expected_box_dim, expected_box_dim), n_holes=(1, 1), p=0.8)

    def test_xtra_tfms_returns_list(self):
        tfms = self.initialise.xtra_tfms()
        self.assertIsInstance(tfms, list)
        for tfm in tfms:
            # Since they are results of calling a MagicMock (the partial),
            # they should be MagicMock objects
            self.assertIsInstance(tfm, MagicMock)

if __name__ == '__main__':
    unittest.main()
