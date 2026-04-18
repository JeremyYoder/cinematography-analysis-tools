import sys
import unittest

class TestInitialise(unittest.TestCase):
    def test_xtra_tfms_default(self):
        import initialise
        tfms = initialise.xtra_tfms()
        self.assertEqual(len(tfms), 3)
        self.assertEqual(tfms[0].__class__.__name__, 'RandomErasing')
        self.assertEqual(tfms[1].__class__.__name__, 'RandomPerspective')
        self.assertEqual(tfms[2].__class__.__name__, 'RandomAffine')

    def test_xtra_tfms_returns_list(self):
        import initialise
        tfms = initialise.xtra_tfms()
        self.assertIsInstance(tfms, list)

    def test_get_tfms(self):
        import initialise
        tfms = initialise.get_tfms()
        self.assertIsInstance(tfms, tuple)
        self.assertEqual(len(tfms), 2)
        train_tfms, valid_tfms = tfms
        self.assertEqual(train_tfms.__class__.__name__, 'Compose')
        self.assertEqual(valid_tfms.__class__.__name__, 'Compose')

if __name__ == '__main__':
    unittest.main()
