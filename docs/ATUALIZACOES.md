# Atualizações Recentes - LibrIA

## Data

2026-03-12

## Escopo desta atualização

Sincronização ampla da documentação com o fluxo embedded atual, export para Pico e organização por diagramas menores.

## Principais mudanças documentadas

### Arquitetura visual
- Criação de `docs/ARCHITECTURE.md`
- Separação do projeto em diagramas menores por etapa
- Inclusão de links de continuidade logo abaixo de cada diagrama para navegação limpa

### Pipeline embedded
- Documentação dos comandos `make train-embedded-all`, `make export-embedded` e `make infer-embedded`
- Registro do bundle em `model/embedded_bundle/`
- Registro do pacote exportável para Pico em `model/embedded_bundle/pico_package/`

### Guias principais revisados
- README principal ajustado para refletir host + embedded
- `docs/DEVELOPMENT.md` reescrito para remover trechos antigos e refletir os comandos reais
- `docs/DATASETS.md` ampliado para cobrir artefatos host, TFLite e bundle final

### Limpeza documental
- Remoção de referências a exemplos C++ experimentais sem integração com o fluxo atual
- Atualização dos índices para apontar para a nova documentação de arquitetura

## Arquivos atualizados nesta rodada

- `README.md`
- `DOCUMENTATION_INDEX.md`
- `docs/README.md`
- `docs/ARCHITECTURE.md`
- `docs/DEVELOPMENT.md`
- `docs/DATASETS.md`
- `docs/ATUALIZACOES.md`
- `CHANGELOG.md`

## Resultado esperado

- O leitor consegue navegar pelo projeto em etapas curtas, sem depender de um diagrama gigante
- A documentação passa a refletir o fluxo atual do repositório, incluindo embedded e export para Pico
- O caminho entre coleta, treino, inferência, bundle e runtime fica explícito e didático
