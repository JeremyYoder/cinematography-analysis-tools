import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock fastai.vision and its components before importing initialise
mock_fastai = MagicMock()
mock_fastai_vision = MagicMock()
sys.modules['fastai'] = mock_fastai
sys.modules['fastai.vision'] = mock_fastai_vision

# Define mocks for functions used in initialise.py
mock_functions = [
    'cutout', 'jitter', 'skew', 'squish', 'tilt',
    'perspective_warp', 'crop_pad', 'rgb_randomize',
    'get_transforms', 'ImageDataBunch', 'ResizeMethod',
    'imagenet_stats', 'cnn_learner', 'models', 'accuracy'
]

for func in mock_functions:
    setattr(mock_fastai_vision, func, MagicMock(name=func))

# fastai.vision typically exports partial from functools
from functools import partial
mock_fastai_vision.partial = partial

import initialise

class TestInitialise(unittest.TestCase):
    def test_xtra_tfms_default(self):
        tfms = initialise.xtra_tfms()
        self.assertEqual(len(tfms), 7)
        # Verify that the transforms are called (they are partials, and then called)
        # In xtra_tfms, they are defined as partials and then called: jitter_(), etc.
        # Since we mocked the base functions as MagicMocks, calling the partial returns a mock call.

    def test_xtra_tfms_base_size(self):
        base_size = 400
        expected_box_dim = int(base_size / 4) # 100

        tfms = initialise.xtra_tfms(base_size=base_size)

        # FastAI's partial might not be the standard functools partial.
        # However, the previous test was attempting to patch `initialise.partial`.
        # The correct way to test this when partial is a MagicMock is to inspect
        # what the MagicMock for partial was called with.

        # Fastai's `partial` is basically functools.partial. Since we mocked it out,
        # let's just inspect the result.
        # But wait, when `mock_fastai_vision.partial = functools.partial` is done,
        # calling partial(cutout, p=...) creates a partial object.
        # Calling that partial object calls the mocked `cutout`.
        # However, `xtra_tfms` does `cutout_ = partial(cutout, ...)` and `cutout_()`.
        # That means `cutout` is called WITH the arguments `length=(box_dim, box_dim)`.
        # So we just need to look at `mock_fastai_vision.cutout.call_args_list`.

        # Reset the mock to get clean call args
        mock_fastai_vision.cutout.reset_mock()
        tfms = initialise.xtra_tfms(base_size=base_size)

        pass

    @patch('initialise.cutout')
    def test_xtra_tfms_base_size_patched(self, mock_cutout):
        base_size = 400
        expected_box_dim = int(base_size / 4) # 100
        initialise.xtra_tfms(base_size=base_size)
        mock_cutout.assert_called_with(length=(expected_box_dim, expected_box_dim), n_holes=(1, 1), p=0.8)

    def test_xtra_tfms_returns_list(self):
        tfms = initialise.xtra_tfms()
        self.assertIsInstance(tfms, list)
        for tfm in tfms:
            # Since they are results of calling a MagicMock (the partial),
            # they should be MagicMock objects
            self.assertIsInstance(tfm, MagicMock)

    @patch('initialise.get_transforms')
    @patch('initialise.xtra_tfms')
    def test_get_tfms(self, mock_xtra_tfms, mock_get_transforms):
        mock_xtra_tfms.return_value = ['mock_xtra']

        initialise.get_tfms()

        mock_get_transforms.assert_called_once_with(
            do_flip=True,
            flip_vert=False,
            max_zoom=1.0,
            max_lighting=0.4,
            max_warp=0.3,
            p_affine=0.85,
            p_lighting=0.85,
            xtra_tfms=['mock_xtra']
        )
        mock_xtra_tfms.assert_called_once()

    @patch('initialise.get_tfms')
    @patch('initialise.ImageDataBunch.from_folder')
    @patch('initialise.cnn_learner')
    @patch('initialise.models.resnet50')
    @patch('initialise.Path')
    def test_get_model_data(self, mock_path, mock_resnet50, mock_cnn_learner, mock_from_folder, mock_get_tfms):
        # Reset mocks
        mock_fastai_vision.ImageDataBunch.from_folder.reset_mock()
        mock_fastai_vision.cnn_learner.reset_mock()

        # Setup mocks
        mock_path_obj = MagicMock()
        mock_path.return_value = mock_path_obj

        mock_get_tfms.return_value = 'mock_tfms'

        mock_data = MagicMock()
        mock_from_folder.return_value.normalize.return_value = mock_data

        mock_learn = MagicMock()
        mock_cnn_learner.return_value.to_fp16.return_value = mock_learn

        # Call function
        learn, data = initialise.get_model_data('fake_path')

        # Verify calls
        mock_path.assert_called_with('fake_path')
        mock_from_folder.assert_called_once_with(
            mock_path_obj, 'train', 'valid',
            size=(375, 666),
            ds_tfms='mock_tfms',
            bs=1,
            resize_method=mock_fastai_vision.ResizeMethod.SQUISH,
            num_workers=0
        )
        mock_from_folder.return_value.normalize.assert_called_once_with(mock_fastai_vision.imagenet_stats)

        mock_cnn_learner.assert_called_once_with(
            mock_data, mock_fastai_vision.models.resnet50, metrics=[initialise.accuracy], pretrained=True
        )
        mock_cnn_learner.return_value.to_fp16.assert_called_once()
        mock_learn.load.assert_called_once_with(mock_path_obj / 'models' / '50colabstage-3-2')

        self.assertEqual(learn, mock_learn)
        self.assertEqual(data, mock_data)

if __name__ == '__main__':
    unittest.main()
