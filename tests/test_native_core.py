"""Compila o núcleo nativo (C++) e roda os testes de paridade com as fixtures.

Pulado quando não há CMake ou compilador C++. A versão com sanitizers e
-Werror fica no ``make native-test``; aqui o build é Release, só para acusar
divergência de comportamento sem pesar na suíte.
"""

import os
import shutil
import subprocess
import tempfile
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_DIR = os.path.join(PROJECT_ROOT, 'native', 'libria_core')

_HAS_TOOLCHAIN = shutil.which('cmake') is not None and any(
    shutil.which(compiler) for compiler in ('c++', 'g++', 'clang++')
)


@unittest.skipUnless(_HAS_TOOLCHAIN, 'CMake ou compilador C++ ausente')
class TestNativeCoreParity(unittest.TestCase):
    def test_native_core_matches_python_fixtures(self):
        with tempfile.TemporaryDirectory() as build_dir:
            steps = [
                ['cmake', '-S', CORE_DIR, '-B', build_dir, '-DCMAKE_BUILD_TYPE=Release'],
                ['cmake', '--build', build_dir],
                ['ctest', '--output-on-failure'],
            ]
            for step in steps:
                result = subprocess.run(
                    step, cwd=build_dir, capture_output=True, text=True, timeout=600, check=False,
                )
                self.assertEqual(
                    result.returncode, 0,
                    f'falhou: {" ".join(step)}\n{result.stdout[-3000:]}\n{result.stderr[-2000:]}',
                )


if __name__ == '__main__':
    unittest.main()
