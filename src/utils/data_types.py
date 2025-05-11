from src.utils.imports import (
    np, # import numpy as np
    TypeVar,
    Generic,
    Union,
    Iterable
) # -> typing module 

__all__ = ['ArrayLike']

_Array_T = TypeVar(
    'T', 
    int, 
    float, 
    list, 
    np.ndarray
) 

class ArrayLike(Generic[_Array_T]):
    """
    Uma classe para representar tipos de dados similares a arrays (como listas, arrays NumPy e valores numéricos).

    A classe `ArrayLike` permite armazenar valores numéricos, listas ou arrays NumPy e oferece funcionalidades convenientes para manipular e acessar esses dados.

    Parameters
    ----------
    dtype : Union[int, float, list, np.ndarray]
        O valor a ser armazenado no objeto `ArrayLike`. Pode ser um valor numérico (`int` ou `float`), uma lista de valores, ou um `np.ndarray`.

    Methods
    -------
    __repr__() -> str
        Retorna uma representação amigável do objeto `ArrayLike`.
    __getitem__(index) -> T
        Permite acessar elementos do array utilizando colchetes (`[]`).
    __len__() -> int
        Retorna o comprimento do array ou 1 se for um valor numérico único.
    to_list() -> list
        Converte o conteúdo do objeto `ArrayLike` para uma lista.
    to_numpy() -> np.ndarray
        Converte o conteúdo do objeto `ArrayLike` para um array NumPy.
    is_numeric() -> bool
        Verifica se o conteúdo do objeto `ArrayLike` é de tipo numérico (`int`, `float`, lista de valores numéricos ou `np.ndarray`).

    Examples
    --------
    >>> array = ArrayLike([1, 2, 3, 4])
    >>> print(array)
    ArrayLike(dtype=[1 2 3 4])
    >>> print(array[2])
    3
    >>> print(len(array))
    4
    >>> print(array.to_list())
    [1, 2, 3, 4]
    >>> print(array.to_numpy())
    [1 2 3 4]
    >>> print(array.is_numeric())
    True
    
    """

    def __init__(self, dtype: Union[_Array_T, Iterable[_Array_T]]) -> None:
        # Inicializa o tipo e transforma em array caso seja um Iterable
        if isinstance(dtype, Iterable) and not isinstance(dtype, (str, bytes)):
            self._dtype = np.array(dtype)
        else:
            self._dtype = dtype

    def __repr__(self) -> str:
        """Retorna uma representação amigável da classe."""
        return f"ArrayLike(dtype={self._dtype})"

    def __getitem__(self, index) -> _Array_T:
        """Permite acessar elementos do array usando colchetes."""
        if isinstance(self._dtype, (np.ndarray, list)):
            return self._dtype[index]
        else:
            raise TypeError("This instance does not support indexing.")

    def __len__(self) -> int:
        """Retorna o comprimento do array se for um array ou iterable."""
        if isinstance(self._dtype, (np.ndarray, list)):
            return len(self._dtype)
        else:
            return 1  # Se for um único valor, o comprimento é 1

    def to_list(self) -> list:
        """Converte o conteúdo em uma lista, caso seja possível."""
        if isinstance(self._dtype, (np.ndarray, list)):
            return self._dtype.tolist() if isinstance(self._dtype, np.ndarray) else list(self._dtype)
        return [self._dtype]

    def to_numpy(self) -> np.ndarray:
        """Converte o conteúdo em um ndarray do NumPy."""
        return np.array(self._dtype)

    def is_numeric(self) -> bool:
        """Verifica se o conteúdo é de um tipo numérico (int ou float)."""
        return isinstance(self._dtype, (int, float, np.ndarray, list))

