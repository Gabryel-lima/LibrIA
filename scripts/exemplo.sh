#!/bin/bash

# Função: subString()
#
# Extrai o texto que está entre dois tokens
#
# Argumento 1 ($1): Texto de entrada
# Argumento 2 ($2): Token inicial
# Argumento 3 ($3): Token final
# Retorna em sucesso: A string entre os dois tokens.
#         Em caso de erro: String vazia se os tokens não forem encontrados
#
# Exemplo de uso:
#   THE_STRING="The thing about windows is that it's a wonderful operating system dude!"
#   RESULT=$(subString "$THE_STRING" "thing about " " is that it")
#   echo "${RESULT}, does not have a good shell"

subString(){
    if [ $# -ne 3 ]; then
        echo "Número inválido de parâmetros passados para $FUNCNAME"
        $(return >/dev/null 2>&1) && return 1 || exit 1
    fi
    echo "$1" | grep -o -P "(?<=$2).*(?=$3)"
}

# Exemplo de uso:
THE_STRING="The thing about windows is that it's a wonderful operating system dude!"
RESULT=$(subString "$THE_STRING" "thing about " " is that it")
echo "${RESULT}, does not have a good shell"
