# Atualizações Recentes - LibrIA

## Data

2026-03-17

## Escopo desta atualização

Sincronização documental com as mudanças recentes em coleta, treino híbrido, persistência de plots e automação por CLI.

## Principais mudanças documentadas

### Espelhamento no dataset
- Registro dos arquivos `sample_XXX_mirror.npy` e `seq_XXX_mirror.npy` como parte oficial do pipeline atual
- Documentação do backfill estático a partir de `frame_XXX.png` legados
- Explicação de como o treino evita duplicar arquivos que já usam o sufixo `_mirror`

### Treino e artefatos visuais
- Inclusão dos gráficos `training_history_*.png` como saídas persistidas dos trainers
- Documentação do diretório `training_plots/accuracy/` e da retenção dos 10 PNGs mais recentes
- Registro do fallback para dataset temporal legado nos trainers temporais

### Inferência e automação
- Atualização do fluxo híbrido para explicitar a comparação entre hipóteses originais e espelhadas
- Registro de que `main.py` retorna exit code `0` em sucesso e `1` em falha
- Ajuste das instruções de desenvolvimento para testes e automações recentes

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

- O leitor entende que o dataset atual inclui variantes espelhadas e que elas participam do treino e da inferência híbrida
- O caminho entre coleta, treino, plots e automação por CLI fica explícito
- Scripts externos conseguem confiar no exit code do `main.py` sem depender de parsing textual
