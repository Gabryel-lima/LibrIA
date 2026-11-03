# Instruções para agentes (Claude e afins)

## To-do e histórico — obrigatório

O to-do do projeto é o [ROADMAP.md](ROADMAP.md). Ele lista só o que **falta**.
Sempre que implementar, corrigir ou concluir algo que esteja nele:

1. **Remova** o item concluído do `ROADMAP.md`. Se ficou parcial, mantenha só o
   que falta e acrescente um link `→ [H-NNN](HISTORICO.md#<âncora>)`.
2. **Registre** no [HISTORICO.md](HISTORICO.md) uma entrada nova **no topo** da
   lista, com o próximo número `H-NNN`, seguindo o modelo do arquivo: data,
   origem, o que mudou, por quê, arquivos, verificação, limites e pendências.
3. **Ligue** a entrada ao código (hash do commit, branch ou PR). Um jeito simples:
   commitar o código primeiro e, num segundo commit, o histórico citando o hash.
4. Se a mudança afeta quem usa o projeto, resuma também no `CHANGELOG.md`.
5. Não apague entradas antigas do histórico; corrija com nota datada.
6. Seja honesto nos limites: diga o que não foi testado (ex.: sem webcam, sem
   TensorFlow, sem GPU) em vez de marcar como concluído.

Correções ou melhorias encontradas pelo caminho, fora do roadmap, também ganham
entrada no histórico.

## Ambiente e testes

- Documentação e mensagens em português.
- Python 3.11 ou 3.12 para o ambiente completo (TensorFlow 2.19.1, MediaPipe
  0.10.14, numpy 1.26.4). Sem eles, a maior parte da suíte roda mesmo assim.
- `make test` (unittest) ou `pytest tests -q`; `make test-coverage` mede a cobertura.
- `make native-test` compila e testa o núcleo C++ (`native/libria_core`).
- Importar `src.inference` carrega o TensorFlow; módulos usados pela coleta e
  pela ingestão devem ficar fora desse pacote (ex.: `utils/`).
