# Plano Arquitetural — Reconhecimento e Tradução de Libras

## Análise do estado atual

* O projeto já possui:

  * Classificação estática das letras A–Y com Random Forest/CNN.
  * Classificação temporal de `J` e `Z` com LSTM/CNN temporal.
  * Extração de landmarks via MediaPipe.
  * Inferência híbrida com janela temporal, confiança e arbitragem.
  * Exportação para TFLite INT8 e runtime C++/Pico.
  * Dataset com suporte a espelhamento e modos `wrist_relative`/`bounding_box`.
* O Transformer existente é experimental e ainda não está integrado ao pipeline.
* O sistema atual reconhece principalmente unidades isoladas — letras e alguns sinais dinâmicos —, não palavras, frases ou contexto linguístico.
* O maior desafio para uso real não é apenas melhorar a acurácia do classificador, mas modelar:

  * Segmentação entre sinais.
  * Sequências de sinais.
  * Expressões faciais e postura corporal.
  * Gramática própria da Libras.
  * Conversão do resultado visual em português natural.
* O bundle TFLite já é uma boa base para Android, iOS e Desktop, mas o contrato de extração de landmarks precisa ser reproduzido de forma idêntica em todas as plataformas.

## Objetivo arquitetural

Separar o produto em quatro camadas:

1. **Percepção visual**

   * Câmera.
   * Detecção de mãos, pose e face.
   * Extração e normalização dos landmarks.

2. **Reconhecimento de sinais**

   * Classificação de sinais isolados.
   * Reconhecimento temporal de palavras ou unidades gestuais.
   * Detecção de início e fim de cada sinal.

3. **Composição linguística**

   * Conversão de sinais reconhecidos em tokens.
   * Formação de palavras e sentenças.
   * Reordenação e normalização para português.
   * Controle de confiança e possibilidade de correção manual.

4. **Apresentação**

   * Texto em tempo real.
   * Histórico da tradução.
   * Reprodução de áudio futuramente.
   * Feedback visual de confiança e sinais não reconhecidos.

## Fases recomendadas

### Fase 1 — Consolidar o reconhecimento de sinais

* Definir claramente o vocabulário inicial de Libras.
* Separar:

  * Alfabeto manual.
  * Sinais lexicais de palavras.
  * Gestos funcionais, como espaço, pausa, apagar e confirmar.
* Ampliar o dataset temporal para palavras completas, não apenas `J` e `Z`.
* Coletar múltiplas pessoas, mãos, velocidades, iluminação, distância e câmeras.
* Registrar metadados por amostra:

  * Pessoa.
  * Câmera.
  * Ambiente.
  * Mão dominante.
  * Duração.
  * Classe.
  * Qualidade da captura.
* Dividir treino, validação e teste por pessoa, evitando que o mesmo usuário apareça nos três conjuntos.
* Medir precisão, recall, matriz de confusão, latência e taxa de rejeição por classe.
* Criar uma classe explícita de **"não reconhecido"** ou **"fora do vocabulário"**.

### Fase 2 — Reconhecimento temporal robusto

* Substituir o reconhecimento baseado somente em janela fixa por um pipeline com:

  * Buffer temporal.
  * Detecção de movimento.
  * Detecção de início e fim do sinal.
  * Suavização das predições.
  * Supressão de duplicatas.
* Comparar LSTM, CNN temporal e Transformer leve usando o mesmo protocolo.
* Priorizar modelos pequenos e quantizáveis para execução local.
* Incorporar landmarks de:

  * Duas mãos.
  * Pose corporal.
  * Face e expressões não manuais.
* Manter o modelo estático como fallback para sinais simples.
* Produzir uma saída padronizada, por exemplo:

  * Token.
  * Classe.
  * Confiança.
  * Tempo inicial e final.
  * Tipo de sinal.
  * Estado de finalização.

### Fase 3 — Formação de palavras e objetos

* Criar um vocabulário de sinais lexicais, começando por um conjunto pequeno e útil.
* Treinar cada palavra como sequência temporal completa, em vez de tentar montar todas as palavras a partir de letras.
* Tratar soletração manual como um modo separado para nomes próprios, termos desconhecidos e palavras novas.
* Criar um banco semântico de:

  * Sinal reconhecido.
  * Palavra em português.
  * Sinônimos.
  * Categoria.
  * Variações regionais.
* Usar contexto para reduzir ambiguidades entre sinais visualmente semelhantes.
* Adicionar mecanismos de confirmação, correção e repetição para baixa confiança.

### Fase 4 — Tradução para frases

* Não conectar diretamente a saída do classificador a um tradutor genérico.
* Criar primeiro uma sequência de tokens de Libras.
* Implementar uma camada linguística capaz de:

  * Agrupar tokens.
  * Detectar pausas.
  * Identificar perguntas, negações e afirmações.
  * Preservar contexto.
  * Converter a ordem típica da Libras para português compreensível.
* Começar com regras e templates controlados.
* Evoluir posteriormente para um modelo seq2seq ou Transformer treinado com pares:

  * Sequência de sinais.
  * Frase em português.
* Avaliar a tradução com métricas linguísticas e revisão de usuários surdos, não apenas acurácia de classificação.

### Fase 5 — Portabilidade para Android, iOS e Desktop

* Definir um contrato único de entrada:

  * Landmarks normalizados.
  * Ordem dos pontos.
  * Sistema de coordenadas.
  * Tratamento de ausência de mão.
  * Tamanho das janelas.
  * Versão do modelo.
* Exportar os modelos finais para TFLite INT8 com manifesto contendo:

  * Labels.
  * Shapes.
  * Feature mode.
  * Versão.
  * Limiares.
  * Classes temporais.
* Criar uma biblioteca de inferência compartilhada, preferencialmente em C++ ou Rust, para evitar implementações divergentes.
* Expor uma API estável para a interface:

  * `loadModel`
  * `processFrame`
  * `pushLandmarks`
  * `predict`
  * `resetSequence`
  * `getTranslation`
* Estratégia sugerida:

  * **Android:** Kotlin/Jetpack CameraX + MediaPipe/TFLite.
  * **iOS:** Swift/AVFoundation + MediaPipe/TFLite.
  * **Desktop:** camada nativa ou aplicação multiplataforma usando o mesmo runtime.
  * **Interface compartilhada:** Flutter pode ser usado para telas e estado, mantendo câmera e inferência em plugins nativos.
* Evitar depender do Python no aplicativo final.
* Garantir execução offline por privacidade, latência e funcionamento sem internet.
* Manter uma API opcional no servidor apenas para telemetria consentida, atualização de modelos e recursos avançados.

### Fase 6 — Produto e experiência de uso

* Criar uma tela de tradução com:

  * Preview da câmera.
  * Texto parcial.
  * Texto confirmado.
  * Confiança.
  * Indicador de processamento.
  * Botões de pausar, apagar e corrigir.
* Permitir selecionar:

  * Alfabeto.
  * Vocabulário.
  * Região ou variante.
  * Modo soletração.
* Adicionar onboarding para posicionamento correto da câmera e iluminação.
* Exibir claramente quando o sistema não tiver confiança suficiente.
* Evitar apresentar uma tradução incorreta como certeza.
* Criar modo de acessibilidade com fonte ampliada, alto contraste e suporte a leitor de tela.

### Fase 7 — Áudio e expansão futura

* Implementar síntese de voz somente após a saída textual estar estável.
* Usar APIs nativas:

  * Android Text-to-Speech.
  * iOS AVSpeechSynthesizer.
  * Desktop equivalente.
* Permitir controlar velocidade, voz e idioma.
* Planejar futuramente a tradução inversa:

  * Texto/voz para Libras.
  * Geração de avatar ou vídeo sinalizado.
* Validar o produto continuamente com pessoas surdas, intérpretes e especialistas em Libras.

## Ordem técnica prioritária

1. Padronizar o contrato de landmarks e metadados.
2. Criar avaliação por pessoa e por ambiente.
3. Expandir o dataset para palavras completas.
4. Implementar segmentação e controle temporal.
5. Adicionar duas mãos, pose e expressões faciais.
6. Criar a saída intermediária de tokens.
7. Implementar a camada de composição linguística.
8. Comparar e quantizar LSTM, CNN temporal e Transformer.
9. Consolidar runtime TFLite multiplataforma.
10. Criar protótipo mobile offline.
11. Validar com usuários reais.
12. Adicionar áudio após estabilizar texto e confiança.

## Critérios de sucesso

* Métricas separadas por pessoa, classe e ambiente.
* Latência compatível com tempo real no dispositivo.
* Funcionamento offline.
* Modelo rejeitando sinais desconhecidos.
* Tradução não duplicando sinais durante uma mesma execução.
* Redução mensurável de erros em ambientes reais.
* Avaliação qualitativa feita por usuários fluentes em Libras.
* Mesma saída para Android, iOS e Desktop usando o mesmo contrato e artefatos de modelo.
