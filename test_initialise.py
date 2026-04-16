import sys
import unittest
import torchvision.transforms as T

import initialise

class TestInitialise(unittest.TestCase):
    def test_xtra_tfms_default(self):
        tfms = initialise.xtra_tfms()
        self.assertEqual(len(tfms), 3)
        self.assertIsInstance(tfms[0], T.RandomErasing)
        self.assertIsInstance(tfms[1], T.RandomPerspective)
        self.assertIsInstance(tfms[2], T.RandomAffine)

    def test_xtra_tfms_returns_list(self):
        tfms = initialise.xtra_tfms()
        self.assertIsInstance(tfms, list)

if __name__ == '__main__':
    unittest.main()
