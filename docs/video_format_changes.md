# Mudanças no Formato de Vídeo - LibrIA

## Resumo das Alterações

O sistema LibrIA foi atualizado para usar o formato **MP4** como padrão para gravação de vídeos, substituindo o formato anterior **AVI**.

## Arquivos Modificados

### 1. `config/settings.py`
- **Alteração**: `output_video_path` mudou de `'output.avi'` para `'output.mp4'`
- **Linha**: 53

### 2. `utils/helpers.py`
- **Função**: `setup_video_recording()`
- **Alteração**: Adicionada lógica para detectar automaticamente o formato baseado na extensão do arquivo
- **Codecs suportados**:
  - `.mp4` → `mp4v`
  - `.avi` → `XVID`
  - Padrão → `mp4v`

### 3. `src/inference/libras_realtime_classifier.py`
- **Função**: `start_classification()`
- **Alteração**: Parâmetro padrão mudou de `'output.avi'` para `'output.mp4'`
- **Função**: `_setup_video_recording()`
- **Alteração**: Mesma lógica de detecção automática de formato

### 4. `backup_old_files/inference_classifier.py`
- **Alteração**: Atualizado para usar MP4 como padrão

## Vantagens do Formato MP4

1. **Melhor Compressão**: Arquivos menores com mesma qualidade
2. **Compatibilidade**: Suportado pela maioria dos players de vídeo
3. **Qualidade**: Melhor relação qualidade/tamanho
4. **Padrão Web**: Formato padrão para streaming e web

## Como Usar

### Gravação Automática (Padrão)
```python
classifier = LibrasRealtimeClassifier()
classifier.start_classification(record_video=True)  # Salva como output.mp4
```

### Gravação com Nome Personalizado
```python
classifier.start_classification(record_video=True, output_path='meu_video.mp4')
```

### Gravação em Formato AVI (Compatibilidade)
```python
classifier.start_classification(record_video=True, output_path='video.avi')
```

## Compatibilidade

O sistema agora suporta automaticamente:
- **MP4** (padrão) - Codec `mp4v`
- **AVI** - Codec `XVID`

A extensão do arquivo determina automaticamente qual codec usar.

## Notas Técnicas

- **Codec MP4**: `mp4v` (H.264/AVC)
- **Codec AVI**: `XVID` (mantido para compatibilidade)
- **FPS**: 20 frames por segundo (configurável)
- **Qualidade**: Mantida a mesma resolução da webcam

## Verificação

Para verificar se as mudanças funcionaram:

1. Execute a inferência:
   ```bash
   python main.py infer
   ```

2. Pressione `r` para iniciar a gravação
3. Verifique se o arquivo `output.mp4` foi criado
4. Confirme que o vídeo pode ser reproduzido normalmente
