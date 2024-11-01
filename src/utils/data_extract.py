import cv2
import pandas as pd
import os

def image_to_csv(image_folder, output_csv):
    """
    Converte uma pasta de imagens em um arquivo CSV com valores de pixels.
    
    Args:
    - image_folder: Caminho da pasta contendo as imagens.
    - output_csv: Caminho completo para o arquivo CSV de saída.
    """
    data = []
    
    # Ordenar os arquivos para garantir uma sequência consistente
    image_files = sorted(os.listdir(image_folder))
    
    for i, image_file in enumerate(image_files):
        # Caminho completo da imagem
        image_path = os.path.join(image_folder, image_file)
        
        # Carregar a imagem em escala de cinza
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        # Redimensionar para 28x28 pixels
        image_resized = cv2.resize(image, (28, 28))
        
        # Obter valores de pixel e adicionar o rótulo no início
        pixels = image_resized.flatten()
        row = [i] + pixels.tolist()  # Usar o índice como rótulo
        
        data.append(row)
    
    # Converter a lista de dados em DataFrame e salvar como CSV
    num_pixels = len(data[0]) - 1 if data else 0
    columns = ['label'] + [f'pixel{j+1}' for j in range(num_pixels)]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Coluns: {columns}")
    print(f"Arquivo CSV salvo em: {output_csv}")

# Exemplo de uso:
if __name__ == '__main__':
    image_to_csv(image_folder="E:\\libria\\data\\hand_keypoint_dataset_26k\\hand_keypoint_dataset_26k\\images\\val", output_csv="E:\\libria\\data\\landmarks_hands_test.csv")
