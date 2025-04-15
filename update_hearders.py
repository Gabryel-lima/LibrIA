import os
import datetime

# Ainda não parece ter funcionado mesmo o add no .git/hook/pre-commit.sh ??

# Autor fixo
AUTHOR = "Gabryel-lima"
HOMEPAGE = "https://github.com/Gabryel-lima"

# Caminho do projeto
PROJECT_DIR = "./src"

def update_headers(project_dir):
    today = datetime.date.today().strftime("%Y-%m-%d")
    header_template = f'''"""
@author : {AUTHOR}
@when : {today}
@homepage : {HOMEPAGE}
"""\n\n'''

    for root, _, files in os.walk(project_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)

                # Evita alterar arquivos do próprio Git ou diretórios ocultos
                if '/.git/' in file_path or '/venv/' in file_path or '/.venv/' in file_path:
                    continue

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Atualiza cabeçalho existente ou adiciona novo
                if content.startswith('"""') and "@author" in content:
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if line.strip().startswith("@when"):
                            lines[i] = f"@when : {today}"
                            break
                    updated_content = '\n'.join(lines)
                else:
                    updated_content = header_template + content

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(updated_content)

                print(f"[OK] Atualizado: {file_path}")

if __name__ == "__main__":
    update_headers(PROJECT_DIR)
