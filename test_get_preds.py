import sys
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import os
from pathlib import Path

# Mock fastai and its components before importing get-preds or initialise
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

from functools import partial
mock_fastai_vision.partial = partial

import importlib.util

spec = importlib.util.spec_from_file_location("get_preds", "get-preds.py")
get_preds = importlib.util.module_from_spec(spec)
sys.modules["get_preds"] = get_preds
spec.loader.exec_module(get_preds)

class TestGetPreds(unittest.TestCase):
    @patch('src.cinematography_tools.predict.ensure_directory')
    @patch('src.cinematography_tools.predict.discover_images')
    @patch('src.cinematography_tools.predict.Image.open')
    @patch('pandas.DataFrame.to_csv')
    @patch('src.cinematography_tools.transforms.get_inference_transforms')
    def test_save_preds(self, mock_transforms, mock_to_csv, mock_open_image, mock_discover_images, mock_ensure_directory):
        import torch
        from src.cinematography_tools.predict import predict_batch

        # Setup mock dependencies
        mock_model = MagicMock(spec=torch.nn.Module)
        mock_device = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(device=mock_device)])

        # Mock predictions: probabilities for each class
        mock_preds = MagicMock()
        # Shape matches tensor operations (batch_size, num_classes)
        # We'll use a standard tensor simulating logits which will be softmaxed
        # SHOT_TYPES are CS, ECS, FS, LS, MS. So MS is index 4.
        import numpy as np
        mock_model.return_value = torch.tensor([[0.1, 0.2, 0.15, 0.05, 5.0]])
        mock_model.return_value = torch.tensor([[0.1, 0.2, 0.15, 0.05, 5.0]])

        mock_file1 = MagicMock(spec=Path)
        mock_file1.relative_to.return_value = 'test1.jpg'
        mock_file1.__str__.return_value = '/fake/img/path/test1.jpg'

        mock_file2 = MagicMock(spec=Path)
        mock_file2.relative_to.return_value = 'test2.png'
        mock_file2.__str__.return_value = '/fake/img/path/test2.png'

        mock_discover_images.return_value = [mock_file1, mock_file2]

        mock_img = MagicMock()
        mock_open_image.return_value = mock_img
        mock_img.convert.return_value = mock_img

        mock_tfms = MagicMock()
        mock_transforms.return_value = mock_tfms
        mock_tfms.return_value = MagicMock()
        mock_tfms.return_value.unsqueeze.return_value = MagicMock()
        mock_tfms.return_value.unsqueeze.return_value.to.return_value = MagicMock()

        # Call predict_batch
        path_img = '/fake/img/path'
        df = predict_batch(mock_model, Path(path_img), output_path=Path(path_img) / 'preds.csv')

        # Verify open_image was called for the image files only
        self.assertEqual(mock_open_image.call_count, 2)
        mock_open_image.assert_any_call('/fake/img/path/test1.jpg')
        mock_open_image.assert_any_call('/fake/img/path/test2.png')

        # Verify model was called
        self.assertEqual(mock_model.call_count, 2)

        # Verify to_csv was called once to save preds.csv
        self.assertEqual(mock_to_csv.call_count, 1)
        args, kwargs = mock_to_csv.call_args
        self.assertEqual(args[0], Path(path_img) / 'preds.csv')
        self.assertEqual(kwargs.get('index'), False)

        # Verify df content
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[0]['shot-type'], 'MS')

    @patch('src.cinematography_tools.predict.ensure_directory')
    @patch('src.cinematography_tools.predict.discover_images')
    @patch('src.cinematography_tools.predict.Image.open')
    @patch('pandas.DataFrame.to_csv')
    @patch('src.cinematography_tools.transforms.get_inference_transforms')
    def test_save_preds_with_path_preds(self, mock_transforms, mock_to_csv, mock_open_image, mock_discover_images, mock_ensure_directory):
        import torch
        from src.cinematography_tools.predict import predict_batch

        mock_model = MagicMock(spec=torch.nn.Module)
        mock_device = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(device=mock_device)])

        mock_model.return_value = torch.tensor([[0.1, 0.2, 5.0, 0.15, 0.05]])

        mock_file1 = MagicMock(spec=Path)
        mock_file1.relative_to.return_value = 'test1.jpg'
        mock_file1.__str__.return_value = '/fake/img/path/test1.jpg'
        mock_discover_images.return_value = [mock_file1]

        mock_img = MagicMock()
        mock_open_image.return_value = mock_img
        mock_img.convert.return_value = mock_img

        mock_tfms = MagicMock()
        mock_transforms.return_value = mock_tfms
        mock_tfms.return_value = MagicMock()
        mock_tfms.return_value.unsqueeze.return_value = MagicMock()
        mock_tfms.return_value.unsqueeze.return_value.to.return_value = MagicMock()

        path_img = '/fake/img/path'
        path_preds = '/fake/preds/path'

        predict_batch(mock_model, Path(path_img), output_path=Path(path_preds) / 'preds.csv')

        # Verify ensure_directory was called
        mock_ensure_directory.assert_called_once_with(Path(path_preds))

        # Verify to_csv was called with path_preds
        self.assertEqual(mock_to_csv.call_count, 1)
        args, kwargs = mock_to_csv.call_args
        self.assertEqual(args[0], Path(path_preds) / 'preds.csv')

if __name__ == '__main__':
    unittest.main()
