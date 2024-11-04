import os
import json
import pandas as pd

class CSVMetadataLogger:
    def __init__(self, csv_files, output_json='csv_metadata.json'):
        """
        Inicializa a classe que coleta e registra as informações dos arquivos CSV.
        
        Args:
        - csv_files (list): Lista de caminhos para os arquivos CSV.
        - output_json (str): Caminho para o arquivo JSON onde serão armazenadas as informações coletadas.
        """
        self.csv_files = csv_files
        self.output_json = output_json
        self.metadata = {}

    def collect_metadata(self):
        """
        Coleta as informações sobre cada arquivo CSV, incluindo a quantidade de amostras, número de colunas e nome das colunas.
        """
        for csv_file in self.csv_files:
            if not os.path.exists(csv_file):
                print(f"Arquivo {csv_file} não encontrado.")
                continue
            
            try:
                df = pd.read_csv(csv_file)
                self.metadata[csv_file] = {
                    'num_samples': len(df),
                    'num_columns': len(df.columns),
                    'column_names': df.columns.tolist(),
                }
            except Exception as e:
                print(f"Erro ao ler o arquivo {csv_file}: {e}")

    def save_metadata(self):
        """
        Salva as informações coletadas em um arquivo JSON.
        """
        try:
            with open(self.output_json, 'w') as json_file:
                json.dump(self.metadata, json_file, indent=4)
            print(f"Informações registradas em {self.output_json}")
        except Exception as e:
            print(f"Erro ao salvar o arquivo JSON: {e}")

    def run(self):
        """
        Executa o processo completo de coleta e salvamento das informações dos arquivos CSV.
        """
        self.collect_metadata()
        self.save_metadata()

# Uso Exemplo
if __name__ == '__main__':
    csv_files = [
        "E:\\libria\\data\\signals_train.csv",
        "E:\\libria\\data\\signals_test.csv",
        "E:\\libria\\data\\landmarks_train.csv",
        "E:\\libria\\data\\landmarks_test.csv",
        "E:\\libria\\data\\hands_train.csv",
        "E:\\libria\\data\\hands_test.csv",
    ]
    logger = CSVMetadataLogger(csv_files)
    logger.run()
