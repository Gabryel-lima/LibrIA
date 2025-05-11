from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# Nome do modulo
module_name = "main_test"

# Configuração do módulo
ext_modules = [
    Pybind11Extension(
        module_name,
        ["main_test.cpp"], # Arquivo fonte C++
        include_dirs=[pybind11.get_include()],  # Diretório de include do Pybind11
    ),
]

setup(
    name=module_name,
    version="0.1",
    description="Exemplo de integração C++ com Python e com Pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
