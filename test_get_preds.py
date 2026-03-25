import sys
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
from pathlib import Path
import importlib

class TestGetPreds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock fastai and initialise before importing get-preds safely
        cls.mock_fastai = MagicMock()
        cls.mock_fastai_vision = MagicMock()

        # Keep old modules to restore later
        cls.old_fastai = sys.modules.get('fastai')
        cls.old_fastai_vision = sys.modules.get('fastai.vision')
        cls.old_initialise = sys.modules.get('initialise')

        sys.modules['fastai'] = cls.mock_fastai
        sys.modules['fastai.vision'] = cls.mock_fastai_vision
        sys.modules['initialise'] = MagicMock()

        # Import the script
        cls.get_preds = importlib.import_module('get-preds')
        cls.test_dir = tempfile.mkdtemp()

        # Create some dummy image files
        Path(os.path.join(cls.test_dir, 'img1.jpg')).touch()
        Path(os.path.join(cls.test_dir, 'img2.png')).touch()
        Path(os.path.join(cls.test_dir, 'img3.jpeg')).touch()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.test_dir)
        # Restore sys.modules
        if cls.old_fastai:
            sys.modules['fastai'] = cls.old_fastai
        else:
            del sys.modules['fastai']

        if cls.old_fastai_vision:
            sys.modules['fastai.vision'] = cls.old_fastai_vision
        else:
            del sys.modules['fastai.vision']

        if cls.old_initialise:
            sys.modules['initialise'] = cls.old_initialise
        else:
            del sys.modules['initialise']

    def test_save_preds_vectorized(self):
        # Mock learn and data
        mock_learn = MagicMock()
        mock_data = MagicMock()

        # Expected classes
        mock_data.classes = ['LS', 'FS', 'MS', 'CS', 'ECS']

        # Mock predict to return different probabilities for the images
        def mock_predict(x):
            filename = x
            if 'img1' in filename:
                probs = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
            elif 'img2' in filename:
                probs = np.array([0.1, 0.4, 0.0, 0.4, 0.1])
            else:
                probs = np.array([0.0, 0.0, 0.1, 0.1, 0.8])

            mock_tensor = MagicMock()
            mock_tensor.numpy.return_value = probs
            return (None, None, mock_tensor)

        mock_learn.predict.side_effect = mock_predict

        out_dir = os.path.join(self.test_dir, 'preds')

        with patch('fastai.vision.open_image', side_effect=lambda x: x):
            self.get_preds.save_preds(self.test_dir, mock_learn, mock_data, out_dir)

        preds_csv_path = os.path.join(out_dir, 'preds.csv')
        self.assertTrue(os.path.exists(preds_csv_path))

        df = pd.read_csv(preds_csv_path)
        self.assertEqual(len(df), 3)
        df = df.sort_values('shot').reset_index(drop=True)

        self.assertEqual(df.loc[0, 'shot'], 'img1.jpg')
        self.assertEqual(df.loc[0, 'shot-type'], 'MS')
        self.assertAlmostEqual(df.loc[0, 'prediction'], 60.0)

        self.assertEqual(df.loc[1, 'shot'], 'img2.png')
        self.assertEqual(df.loc[1, 'shot-type'], 'FS')
        self.assertAlmostEqual(df.loc[1, 'prediction'], 40.0)

        self.assertEqual(df.loc[2, 'shot'], 'img3.jpeg')
        self.assertEqual(df.loc[2, 'shot-type'], 'ECS')
        self.assertAlmostEqual(df.loc[2, 'prediction'], 80.0)

if __name__ == '__main__':
    unittest.main()
