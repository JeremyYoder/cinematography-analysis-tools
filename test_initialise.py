import sys
import unittest
import torchvision.transforms as T

# Since initialise just imports from cinematography_tools
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
        for tfm in tfms:
            # Check they are torchvision transforms, not MagicMocks
            self.assertTrue(isinstance(tfm, T.RandomErasing) or
                            isinstance(tfm, T.RandomPerspective) or
                            isinstance(tfm, T.RandomAffine))

if __name__ == '__main__':
    unittest.main()
