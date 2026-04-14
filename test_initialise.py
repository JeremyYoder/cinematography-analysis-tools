import unittest
import initialise
import torchvision.transforms as T

class TestInitialise(unittest.TestCase):
    def test_xtra_tfms_default(self):
        tfms = initialise.xtra_tfms()
        self.assertEqual(len(tfms), 3)

    def test_xtra_tfms_returns_list(self):
        tfms = initialise.xtra_tfms()
        self.assertIsInstance(tfms, list)

        expected_types = (T.RandomErasing, T.RandomPerspective, T.RandomAffine)
        for tfm in tfms:
            self.assertTrue(any(isinstance(tfm, expected_type) for expected_type in expected_types))

if __name__ == '__main__':
    unittest.main()
