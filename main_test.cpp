#include <pybind11/pybind11.h>

int soma(int a, int b) {
    return a + b;
}

int subtrai(int a, int b) {
    return a - b;
}

PYBIND11_MODULE(main_test, m) { // ertifique-se do nome do modulo estar correto no setup.py
    m.def("soma", &soma, "Função para somar dois números");
    m.def("subtrai", &subtrai, "Função para subtrair dois números");
}

