"""Os .md do repositório não podem conter ``{{`` nem ``{%``.

O GitHub Pages (Jekyll, Liquid 4) renderiza todos os .md e trata essas sequências
como template, inclusive dentro de blocos de código. Um exemplo C++ com uma
inicialização aninhada (duas chaves de abertura seguidas) já derrubou o build do
Pages: o log dizia ``Liquid syntax error ... Variable was not properly
terminated``. Escreva as chaves com um espaço entre elas.
"""

import re
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIQUID = re.compile(r'\{\{|\{%')
IGNORED_DIRS = {'.git', '.github', 'node_modules', '.venv', 'build', '.gradle'}


def _markdown_files():
    for path in PROJECT_ROOT.rglob('*.md'):
        relative = path.relative_to(PROJECT_ROOT)
        if not IGNORED_DIRS.intersection(relative.parts):
            yield path, relative


class TestMarkdownIsLiquidSafe(unittest.TestCase):
    def test_no_liquid_delimiters_in_markdown(self):
        offenders = []
        for path, relative in _markdown_files():
            for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
                if LIQUID.search(line):
                    offenders.append(f'{relative}:{number}: {line.strip()[:80]}')
        self.assertEqual(
            offenders, [],
            'Sequências Liquid quebram o build do GitHub Pages; ponha um espaço entre as chaves:\n'
            + '\n'.join(offenders),
        )


if __name__ == '__main__':
    unittest.main()
