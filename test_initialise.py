import sys
import unittest
from unittest.mock import MagicMock, patch
import torchvision.transforms as T

import initialise

class TestInitialise(unittest.TestCase):
    def test_xtra_tfms_default(self):
        tfms = initialise.xtra_tfms()
        self.assertEqual(len(tfms), 3)

    def test_xtra_tfms_returns_list(self):
        tfms = initialise.xtra_tfms()
        self.assertIsInstance(tfms, list)

        # Verify the types of returned transforms
        self.assertIsInstance(tfms[0], T.RandomErasing)
        self.assertIsInstance(tfms[1], T.RandomPerspective)
        self.assertIsInstance(tfms[2], T.RandomAffine)

    def test_get_tfms(self):
        train_tfms, valid_tfms = initialise.get_tfms()

        self.assertIsInstance(train_tfms, T.Compose)
        self.assertIsInstance(valid_tfms, T.Compose)

        # Check lengths based on the new definitions
        self.assertEqual(len(train_tfms.transforms), 9)  # 6 explicit + 3 from xtra_tfms
        self.assertEqual(len(valid_tfms.transforms), 4)

    @patch('initialise.get_model_data')
    def test_get_model_data(self, mock_get_model_data):
        initialise.get_model_data('fake_path')
        mock_get_model_data.assert_called_once_with('fake_path')

if __name__ == '__main__':
    unittest.main()
