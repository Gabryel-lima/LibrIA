#include <pybind11/pybind11.h>

int soma(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(main, m) {
    m.def("soma", &soma, "Função para somar dois numeros");
}


