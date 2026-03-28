import sys
import unittest
from unittest.mock import MagicMock, patch
import os
import importlib

# Mock heavy dependencies globally before importing get-preds
mock_fastai = MagicMock()
mock_fastai_vision = MagicMock()
mock_pandas = MagicMock()

sys.modules['fastai'] = mock_fastai
sys.modules['fastai.vision'] = mock_fastai_vision
sys.modules['pandas'] = mock_pandas

class TestGetPreds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # We use importlib because of the hyphen in get-preds.py
        cls.get_preds = importlib.import_module('get-preds')

    def test_save_preds_logic(self):
        with patch.object(self.get_preds.os, 'listdir') as mock_listdir, \
             patch.object(self.get_preds.os.path, 'exists') as mock_exists, \
             patch.object(self.get_preds.os, 'mkdir') as mock_mkdir:

            mock_exists.return_value = True
            mock_listdir.return_value = ['img1.jpg']

            # We need to mock open_image which we explicitly imported
            self.get_preds.open_image = MagicMock(return_value='mocked_image')

            # Mock learn and data
            mock_learn = MagicMock()
            mock_data = MagicMock()
            mock_data.classes = ['CS', 'ECS', 'FS', 'LS', 'MS']

            mock_res = MagicMock()
            mock_res.numpy.return_value = [0.1, 0.05, 0.2, 0.6, 0.05]

            def mock_predict(img):
                return [None, None, mock_res]

            mock_learn.predict.side_effect = mock_predict

            # We need to mock pandas DataFrame
            mock_df_instance = MagicMock()

            # We need to mock columns correctly, so that `available_cats = [c for c in categories if c in df.columns]` works
            mock_df_instance.columns = ['CS', 'ECS', 'FS', 'LS', 'MS', 'shot']

            # Mock indexing. df[available_cats] returns another mock
            mock_sub_df = MagicMock()

            # __getitem__ mock for df[available_cats]
            def getitem_side_effect(key):
                if isinstance(key, list):
                    return mock_sub_df
                return MagicMock()
            mock_df_instance.__getitem__.side_effect = getitem_side_effect

            # Mock math operations
            mock_sub_df.__mul__.return_value = mock_sub_df

            # Mock max and idxmax
            mock_sub_df.max.return_value = [60.0]
            mock_sub_df.idxmax.return_value = ['LS']

            # Also df[['shot-type', 'prediction', 'shot']]
            mock_final_df = MagicMock()
            def getitem_final(key):
                if key == ['shot-type', 'prediction', 'shot']:
                    return mock_final_df
                if isinstance(key, list) and 'LS' in key:
                    return mock_sub_df
                return MagicMock()
            mock_df_instance.__getitem__.side_effect = getitem_final

            # Resetting side effect
            def setitem_side_effect(key, value):
                pass
            mock_df_instance.__setitem__.side_effect = setitem_side_effect

            mock_pandas.DataFrame.return_value = mock_df_instance

            self.get_preds.save_preds(mock_learn, mock_data, '/fake/img/path', '/fake/preds/path')

            self.assertTrue(mock_pandas.DataFrame.called)
            self.assertTrue(mock_final_df.to_csv.called)

if __name__ == '__main__':
    unittest.main()
