import unittest
from unittest.mock import patch

import main as libria_main


class MainCommandExitCodeTests(unittest.TestCase):
    def test_run_command_propagates_training_failure(self):
        with patch('main.train_all_models', return_value=False):
            self.assertFalse(libria_main.run_command('train'))

    def test_run_command_propagates_training_success(self):
        with patch('main.train_temporal_model', return_value=True):
            self.assertTrue(libria_main.run_command('train-temporal'))

    def test_unknown_command_fails(self):
        self.assertFalse(libria_main.run_command('comando-inexistente'))

    def test_main_exits_with_nonzero_when_training_fails(self):
        with patch('main.setup_logging', return_value=None), patch(
            'main.train_all_models', return_value=False
        ), patch('sys.argv', ['main.py', 'train']):
            with self.assertRaises(SystemExit) as raised:
                libria_main.main()
            self.assertEqual(raised.exception.code, 1)

    def test_main_exits_zero_when_training_succeeds(self):
        with patch('main.setup_logging', return_value=None), patch(
            'main.train_all_models', return_value=True
        ), patch('sys.argv', ['main.py', 'train']):
            with self.assertRaises(SystemExit) as raised:
                libria_main.main()
            self.assertEqual(raised.exception.code, 0)


class MainCommandParityTests(unittest.TestCase):
    """Os comandos do main.py e os alvos do Makefile devem ter os mesmos nomes."""

    def _makefile_targets(self):
        targets = set()
        with open('Makefile', 'r', encoding='utf-8') as file_obj:
            for line in file_obj:
                if line.startswith(('\t', ' ', '#', '.')) or '##' not in line:
                    continue
                name = line.split(':', 1)[0].strip()
                if name:
                    targets.add(name)
        return targets

    def test_every_pipeline_command_has_a_makefile_target(self):
        targets = self._makefile_targets()
        missing = sorted(
            command for command in libria_main.COMMANDS
            if command not in targets and command != 'all'
        )

        self.assertEqual(missing, [], f'comandos sem alvo no Makefile: {missing}')


if __name__ == '__main__':
    unittest.main()
