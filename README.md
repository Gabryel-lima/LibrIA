<h1>Libria</h1>
<p>O Libria é um projeto de inteligência artificial desenvolvido para traduzir sinais de mão da Língua Brasileira de Sinais (Libras) para texto ou outras formas de comunicação. Utilizando redes neurais baseadas em ResNet e Transformers, o sistema reconhece gestos e interpreta sinais em tempo real, fornecendo feedback visual e textual sobre o processo.</p>

<h2>Objetivos Principais</h2>
<ul>
    <li><strong>Reconhecer gestos</strong> de Libras e classificá-los com alta precisão.</li>
    <li><strong>Identificar landmarks</strong> das mãos e utilizá-los para melhorar a compreensão da estrutura dos sinais.</li>
    <li><strong>Suportar entrada em tempo real</strong> de vídeo, permitindo uma tradução dinâmica e interativa.</li>
    <li><strong>Fornecer explicabilidade</strong> ao modelo, utilizando técnicas como Grad-CAM para destacar as regiões mais relevantes na decisão da rede.</li>
    <li><strong>Implementar processamento temporal</strong> para melhorar a tradução de sequências gestuais complexas.</li>
    <li><strong>Conjunto de dados</strong>
    <a href="https://www.kaggle.com/datasets/grasshoppermouse/libras-dataset">Libras Dataset</a> do Kaggle, com 10.000 imagens de sinais de Libras.</li>
</ul>

<h2>Demonstração</h2>
<video class="demo-video" controls>
  <source src="docs/api/Filter_87.mp4" type="video/mp4">
  Seu navegador não suporta o elemento <code>video</code>.
    <style>
        .demo-video {
            width: 100%;
            height: auto;
        }
    </style>
</video>


<h2>Diferenciais</h2>
<ul>
    <li>Suporte para <strong>múltiplos tipos de entradas</strong> (imagens, landmarks e vídeo).</li>
    <li>Uso de técnicas avançadas de <strong>balanceamento de classes</strong> e aumento de dados para lidar com datasets desbalanceados.</li>
    <li>Interface para <strong>testes ao vivo</strong> com webcams ou dispositivos móveis.</li>
    <li>Desenvolvimento em <strong>Python</strong> com portabilidade planejada para <strong>C++</strong> para execução em diferentes hardwares.</li>
</ul>

<h2>Status Atual</h2>
<ul>
    <li>Estou adaptando outras arquiteturas e testando alternativas.</>
    <li>Modelos baseados em <strong>ResNet, MobileNet e Transformer</strong> parcialmente treinados e testados.</li>
    <li>Implementação de <strong>inferência em tempo real</strong> utilizando MediaPipe para landmarks.</li>
    <li>Ajustes contínuos para melhorar <strong>precisão e tempo de resposta</strong>.</li>
</ul>

<h2>Próximos Passos</h2>
<ul>
    <li><strong>Otimizar o pipeline</strong> de treinamento e inferência.</li>
    <li><strong>Melhorar a representatividade</strong> do dataset com novos exemplos de sinais e landmarks.</li>
    <li><strong>Integrar o modelo a dispositivos embarcados</strong> utilizando C++.</li>
    <li><strong>Estudar alternativas linguagens</strong>, para usuários finais. Junto da criação de um app acessível.</li>
    <li><strong>Desenvolver uma interface gráfica</strong> acessível para usuários finais.</li>
</ul>

<h2>Tecnologias Utilizadas</h2>
<ul>
    <li><strong>Python</strong> (TensorFlow, Keras, MediaPipe)</li>
    <li><strong>C++</strong> (para portabilidade e otimização de hardware)</li>
    <li><strong>Linux</strong> (Ambiente de desenvolvimento)</li>
</ul>

<h2>Fluxograma *Imagem ainda a definir</h2>
<figure>
  <img 
    src="docs/Libria_Fluxograma_para_Educação_e_Acessibilidade_em_Libras_com_IA.jpg" 
    alt="Fluxograma do projeto Libria para educação e acessibilidade em Libras" 
    width="600"
  >
  <figcaption>
    Fluxograma do processo de tradução de sinais de Libras para texto no projeto Libria.
  </figcaption>
</figure>

## License
Este projeto está licenciado sob a [MIT License](LICENSE).
