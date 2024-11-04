import pandas as pd

class CSVValidator:
    def __init__(self, csv_paths):
        """
        Inicializa a classe CSVValidator.
        
        Args:
        - csv_paths (list): Lista com os caminhos dos arquivos CSV a serem validados.
        """
        self.csv_paths = csv_paths
        self.dataframes = [pd.read_csv(path) for path in csv_paths]

    def validate_shapes(self):
        """
        Verifica se todos os arquivos CSV possuem o mesmo número de amostras (linhas).
        
        Returns:
        - dict: Dicionário indicando se os tamanhos batem ou não para cada par de CSVs.
        """
        results = {}
        for i in range(len(self.dataframes)):
            for j in range(i + 1, len(self.dataframes)):
                df1, df2 = self.dataframes[i], self.dataframes[j]
                result_key = f"{self.csv_paths[i]} vs {self.csv_paths[j]}"
                results[result_key] = len(df1) == len(df2)
        return results

    def validate_labels(self):
        """
        Verifica se todos os arquivos CSV possuem os mesmos rótulos (labels) únicos.
        
        Returns:
        - dict: Dicionário indicando se os rótulos batem ou não para cada par de CSVs.
        """
        results = {}
        for i in range(len(self.dataframes)):
            for j in range(i + 1, len(self.dataframes)):
                df1_labels = set(self.dataframes[i]['label'].unique())
                df2_labels = set(self.dataframes[j]['label'].unique())
                result_key = f"{self.csv_paths[i]} vs {self.csv_paths[j]}"
                results[result_key] = df1_labels == df2_labels
        return results

    def validate(self):
        """
        Executa todas as validações (shapes e labels) e retorna os resultados.
        
        Returns:
        - dict: Dicionário contendo os resultados das validações de shapes e labels.
        """
        shape_results = self.validate_shapes()
        label_results = self.validate_labels()
        
        return {
            'shape_validation': shape_results,
            'label_validation': label_results
        }

# Exemplo de uso
if __name__ == "__main__":
    csv_paths = [
        "E:\\libria\\data\\signals_train.csv",
        "E:\\libria\\data\\signals_test.csv",
        "E:\\libria\\data\\landmarks_train.csv",
        "E:\\libria\\data\\landmarks_test.csv",
        "E:\\libria\\data\\hands_train.csv",
        "E:\\libria\\data\\hands_test.csv"
    ]
    
    validator = CSVValidator(csv_paths)
    validation_results = validator.validate()
    
    # Exibir resultados
    print("Validação de Shapes:")
    for key, value in validation_results['shape_validation'].items():
        print(f"{key}: {'Batem' if value else 'Não Batem'}")
    
    print("\nValidação de Labels:")
    for key, value in validation_results['label_validation'].items():
        print(f"{key}: {'Batem' if value else 'Não Batem'}")
