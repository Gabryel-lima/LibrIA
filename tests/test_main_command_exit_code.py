import unittest
from unittest.mock import patch

import main as libria_main


class MainCommandExitCodeTests(unittest.TestCase):
    def test_run_command_propagates_training_failure(self):
        with patch('main.train_hybrid_models', return_value=False):
            self.assertFalse(libria_main.run_command('train_hybrid'))

    def test_run_command_propagates_training_success(self):
        with patch('main.train_lstm_model', return_value=True):
            self.assertTrue(libria_main.run_command('train_lstm'))

    def test_main_exits_with_nonzero_when_training_fails(self):
        with patch('main.setup_logging', return_value=None), patch(
            'main.train_hybrid_models', return_value=False
        ), patch('sys.argv', ['main.py', 'train_hybrid']):
            with self.assertRaises(SystemExit) as raised:
                libria_main.main()
            self.assertEqual(raised.exception.code, 1)

    def test_main_exits_zero_when_training_succeeds(self):
        with patch('main.setup_logging', return_value=None), patch(
            'main.train_hybrid_models', return_value=True
        ), patch('sys.argv', ['main.py', 'train_hybrid']):
            with self.assertRaises(SystemExit) as raised:
                libria_main.main()
            self.assertEqual(raised.exception.code, 0)


if __name__ == '__main__':
    unittest.main()
