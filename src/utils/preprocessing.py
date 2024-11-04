from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def find_outliers_iqr_with_combined_histogram(*dfs: pd.DataFrame, sample_size: int = 1000) -> Dict[str, pd.DataFrame]:
    """
    Identifica outliers usando o método IQR em múltiplos DataFrames e exibe um histograma combinado das colunas.

    Args:
    - *dfs: Múltiplos DataFrames para análise de outliers.
    - sample_size: Tamanho da amostra aleatória para o histograma combinado (default: 1000).

    Returns:
    - Dicionário com os nomes das colunas como chaves e Series de outliers como valores.
    """
    combined_outliers = {}
    combined_df = pd.concat(dfs, ignore_index=True)  # Combina todos os DataFrames

    # Obtém todas as colunas numéricas do DataFrame combinado
    numeric_columns = combined_df.select_dtypes(include=['float64', 'int64']).columns
    combined_data = pd.concat([combined_df[column].dropna() for column in numeric_columns])

    # Calcula o IQR e os limites de outliers com base nos dados combinados
    Q1 = combined_data.quantile(0.25)
    Q3 = combined_data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identifica os outliers nos dados combinados
    combined_outliers = combined_data[(combined_data < lower_bound) | (combined_data > upper_bound)].dropna()

    # Amostra aleatória dos dados combinados para o histograma
    sampled_data = combined_data.sample(min(len(combined_data), sample_size), random_state=42)

    # Plota o histograma combinado
    plt.figure(figsize=(9, 6))
    plt.hist(sampled_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(lower_bound, color='red', linestyle='--', linewidth=2, label='Limite Inferior (Outliers)')
    plt.axvline(upper_bound, color='green', linestyle='--', linewidth=2, label='Limite Superior (Outliers)')
    
    # Adiciona título e rótulos mais detalhados
    plt.title('Histograma Combinado das Colunas Numéricas com Limites de Outliers', fontsize=16)
    plt.xlabel('Valores das Colunas Numéricas', fontsize=14)
    plt.ylabel('Frequência', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12)
    
    # Adiciona uma anotação explicativa sobre os limites de outliers
    plt.text(lower_bound, plt.ylim()[1] * 0.9, 'Limite Inferior', color='red', fontsize=12, rotation=90, ha='right')
    plt.text(upper_bound, plt.ylim()[1] * 0.9, 'Limite Superior', color='green', fontsize=12, rotation=90, ha='left')
    
    # Adiciona uma grade para facilitar a visualização
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()
    
    return combined_outliers

def verificar_normalizacao(data, nome):
    # Calculando média e desvio padrão antes da normalização
    media_antes = np.mean(data, axis=0)
    desvio_padrao_antes = np.std(data, axis=0)
    
    # Exibindo apenas os primeiros 20 valores
    print(f"Média antes de normalizar ({nome}): {media_antes[:20]}")
    print(f"Desvio padrão antes de normalizar ({nome}): {desvio_padrao_antes[:20]}")
    
    # Normalizar os dados
    data_normalizado = (data - media_antes) / (desvio_padrao_antes + 1e-8)
    
    # Calculando média e desvio padrão depois da normalização
    media_depois = np.mean(data_normalizado, axis=0)
    desvio_padrao_depois = np.std(data_normalizado, axis=0)
    
    # Exibindo apenas os primeiros 20 valores
    print(f"Média depois de normalizar ({nome}): {media_depois[:20]}")
    print(f"Desvio padrão depois de normalizar ({nome}): {desvio_padrao_depois[:20]}")
    
    return data_normalizado


