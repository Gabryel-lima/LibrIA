# Binding C++/Python com pybind11 no repositorio

Este repositorio ja possui um exemplo minimo de extensao Python em C++ usando `pybind11`.

## Arquivos envolvidos

- `main_test.cpp`
- `setup.py`

## O que o exemplo faz

`main_test.cpp` expoe duas funcoes simples:

- `soma(int a, int b)`
- `subtrai(int a, int b)`

Elas sao publicadas como modulo Python chamado `main_test`:

```cpp
PYBIND11_MODULE(main_test, m) {
  m.def("soma", &soma, "Funcao para somar dois numeros");
  m.def("subtrai", &subtrai, "Funcao para subtrair dois numeros");
}
```

No `setup.py`, o nome do modulo tambem esta fixado como `main_test`, entao esses dois arquivos precisam continuar sincronizados.

## Como compilar

Com o ambiente virtual ativo:

```bash
pip install pybind11 setuptools wheel
python setup.py build_ext --inplace
```

Isso gera um binario de extensao Python na raiz do projeto, com nome semelhante a:

```text
main_test.cpython-311-x86_64-linux-gnu.so
```

## Como testar

```bash
python -c "import main_test; print(main_test.soma(2, 3)); print(main_test.subtrai(7, 4))"
```

Saida esperada:

```text
5
3
```

## Papel desse exemplo no projeto

Esse binding e apenas demonstrativo. Ele nao participa do pipeline principal de LibrIA, que permanece em Python e usa:

- `main.py`
- `src/data_processing/`
- `src/model_training/`
- `src/inference/`

## Cuidados

1. O nome em `PYBIND11_MODULE(...)` deve ser o mesmo de `module_name` em `setup.py`.
2. O build deve acontecer no mesmo Python/venv em que voce vai importar o modulo.
3. Se o Python da IDE estiver diferente do Python do build, o import vai falhar ou o autocomplete vai ficar inconsistente.

## Uso em IDE

No VS Code, o ponto principal nao e adicionar `extraPaths`, e sim garantir que o interpretador selecionado seja o mesmo da `.venv` onde o modulo foi compilado.

## Possiveis proximos passos

Se esse exemplo evoluir para algo util ao LibrIA, o caminho natural seria:

1. mover o binding para uma pasta dedicada
2. expor preprocessamento numerico ou partes de inferencia de alto custo
3. adicionar um alvo especifico no Makefile para compilar a extensao

Atualizado em 2026-03-10.
