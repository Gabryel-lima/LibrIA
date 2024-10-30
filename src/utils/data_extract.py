import cv2
import pandas as pd
import os

def image_to_csv(image_folder, output_csv, labels):
    """
    Converte uma pasta de imagens em um arquivo CSV com valores de pixels.
    
    Args:
    - image_folder: Caminho da pasta contendo as imagens.
    - output_csv: Caminho do arquivo CSV de saída.
    - labels: Lista de rótulos (deve corresponder ao número de imagens).
    """
    data = []
    
    for i, image_file in enumerate(os.listdir(image_folder)):
        # Carregar a imagem
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Redimensionar para 28x28 pixels
        image_resized = cv2.resize(image, (28, 28))
        
        # Obter valores de pixel e adicionar o rótulo no início
        pixels = image_resized.flatten()
        row = [labels[i]] + pixels.tolist()  # Adicionar rótulo e valores de pixel
        
        data.append(row)
    
    # Converter a lista de dados em DataFrame e salvar como CSV
    columns = ['label'] + [f'pixel{j+1}' for j in range(784)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Arquivo CSV salvo em: {output_csv}")

# Exemplo de uso:
# image_to_csv('caminho/para/pasta/imagens', 'output.csv', labels)
