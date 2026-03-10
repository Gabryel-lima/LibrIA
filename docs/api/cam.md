# API C++ de Camera e Inferencia

Este documento descreve o que o repositorio ja possui em C++ para captura de camera e inferencia com LibTorch.

## Arquivo existente

Implementacao atual:

- `src/interfaces/cam.cpp`

Esse arquivo:

- abre a camera com OpenCV
- carrega um modelo TorchScript de `src/saved/classification.pth`
- converte frames para RGB
- redimensiona para `224x224`
- monta um tensor no formato `NCHW`
- executa `model.forward()` em CPU

## Fluxo atual em `cam.cpp`

Resumo do que o codigo faz hoje:

```cpp
torch::jit::script::Module model = torch::jit::load("src/saved/classification.pth");
cv::VideoCapture cap(0);

cap >> frame;
cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
cv::resize(frame, frame, cv::Size(224, 224));
frame.convertTo(frame, CV_32F, 1.0f / 255.0f);

torch::Tensor input = torch::from_blob(frame.data, {1, 224, 224, 3}, torch::kFloat).clone();
input = input.permute({0, 3, 1, 2}).contiguous();
torch::Tensor output = model.forward({input}).toTensor();
```

## Dependencias necessarias

No Linux, voce precisa de:

- OpenCV com headers e libs acessiveis
- LibTorch compativel com sua toolchain
- um modelo TorchScript valido em `src/saved/classification.pth`

## Exemplo de compilacao manual

O repositorio nao tem hoje um `CMakeLists.txt` ativo para esse binario, entao a forma mais simples de documentar o build e por linha de comando.

Exemplo ilustrativo:

```bash
g++ src/interfaces/cam.cpp -o cam \
  $(pkg-config --cflags --libs opencv4) \
  -I/path/to/libtorch/include \
  -I/path/to/libtorch/include/torch/csrc/api/include \
  -L/path/to/libtorch/lib \
  -ltorch -ltorch_cpu -lc10 \
  -Wl,-rpath,/path/to/libtorch/lib
```

Se o OpenCV estiver instalado pelo sistema, voce pode localizar os includes com:

```bash
find /usr/include -name opencv2 | head
```

## Limitacoes atuais

- o caminho do modelo esta fixo no codigo
- nao ha pos-processamento do `output`
- o programa apenas imprime sucesso/erro da inferencia
- nao ha integracao com a calibracao de camera do pipeline Python
- nao ha sistema de build dedicado para esse executavel dentro do repo

## Melhorias recomendadas

1. Parametrizar camera e caminho do modelo por argumento de linha de comando.
2. Interpretar `output` e exibir classe/confidencia no frame.
3. Reaproveitar o mesmo protocolo de preprocessamento do pipeline Python.
4. Adicionar um build reproducivel com CMake ou Make target especifico.

## Relacao com o restante do projeto

Essa interface C++ e experimental e paralela ao pipeline principal em Python. O fluxo mais completo e mantido hoje em:

- `src/inference/libras_realtime_classifier.py`
- `src/inference/libras_lstm_realtime_classifier.py`
- `scripts/calibrate_camera.py`

Atualizado em 2026-03-10.
