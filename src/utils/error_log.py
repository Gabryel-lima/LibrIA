import traceback

def error_log(file: str = 'error_log'):
    """Função para registrar o erro em um arquivo log."""
    with open(file + '.txt', 'w') as f:
        f.write('An exception occurred:\n')
        f.write(traceback.format_exc())
    print(f'An exception occurred. Check {file} for details.')
