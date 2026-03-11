# Compatibilidade com CPUs sem AVX

## Resumo

Parte do stack do LibrIA depende de bibliotecas que costumam exigir suporte AVX na CPU. Em máquinas antigas, isso pode gerar falhas no import ou erros como `illegal hardware instruction`.

## Bibliotecas mais sensíveis

- TensorFlow 2.16.1
- Keras 3.4.1
- MediaPipe 0.10.14
- Em alguns ambientes, builds específicas de PyTorch também podem exigir AVX

## Impacto prático no projeto

### Fluxos que dependem fortemente de AVX

- `make collect-static` e `python main.py collect_static`
- `make infer` quando a extração de landmarks precisa do MediaPipe
- `make collect-temporal`
- `make train-lstm`
- `make infer-lstm`

### Fluxos que continuam úteis para diagnóstico

- `make environment`
- `make verify-setup`
- `python test_setup.py`
- Operações em arquivos, documentação e revisão de código

## Comportamento atual do projeto

- `test_setup.py` detecta ausência de AVX e emite aviso
- `Makefile` exibe avisos para TensorFlow, MediaPipe e Keras quando não disponíveis
- Alguns módulos fazem import condicional e retornam erro explícito ao tentar usar funcionalidade indisponível

## Como verificar AVX no Linux

```bash
grep -q avx /proc/cpuinfo && echo "AVX disponível" || echo "Sem AVX"
```

## Verificação recomendada do ambiente

```bash
make setup
source .venv/bin/activate
make verify-setup
python test_setup.py
```

## Estratégias recomendadas

### Se você tem uma máquina com AVX

Use essa máquina para:

- coleta com MediaPipe
- treino e inferência da LSTM
- geração de artefatos finais

### Se você não tem AVX

Use a máquina sem AVX para:

- editar código
- revisar documentação
- trabalhar com os artefatos já gerados
- validar partes do projeto que não dependem do stack de visão/deep learning

## Observação importante sobre `requirements.txt`

O arquivo de dependências atual já lista TensorFlow, Keras, PyTorch e MediaPipe. Em CPUs incompatíveis, a instalação ou o uso dessas bibliotecas pode falhar. A recomendação é usar uma máquina com AVX para o setup completo.

## Artefatos portáveis

Mesmo quando o treino é feito em outra máquina, faz sentido versionar ou transferir:

- `model/model.pickle`
- `model/libras_lstm.keras`
- `model/libras_lstm_labels.pickle`
- `config/camera_matrix.npy`
- `config/dist_coeffs.npy`

## Última atualização

2026-03-10
