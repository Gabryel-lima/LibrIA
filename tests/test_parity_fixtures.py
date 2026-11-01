"""As fixtures de paridade precisam acompanhar o pipeline Python.

``native/fixtures/parity_fixtures.json`` é a referência que o núcleo C++ (e, no
futuro, o app Android) precisa reproduzir. Se o comportamento do pipeline
Python mudar e o arquivo não for regenerado, o núcleo nativo passaria a ser
testado contra um comportamento antigo. Este teste impede isso.
"""

import unittest


class TestParityFixturesUpToDate(unittest.TestCase):
    def test_committed_fixtures_match_python_pipeline(self):
        from scripts.export_parity_fixtures import DEFAULT_OUTPUT, build_fixtures, render

        expected = render(build_fixtures())
        current = DEFAULT_OUTPUT.read_text(encoding='utf-8')
        # assertTrue em vez de assertEqual: o diff de ~400 KiB não ajudaria.
        self.assertTrue(
            current == expected,
            f'{DEFAULT_OUTPUT} desatualizado. Rode: python -m scripts.export_parity_fixtures '
            '(ou make native-fixtures) e depois make native-test',
        )


if __name__ == '__main__':
    unittest.main()
