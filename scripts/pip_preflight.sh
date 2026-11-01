#!/usr/bin/env bash
# Pré-voo das instalações pip: limpa o cache e confere se há espaço/cota.
# Uso: scripts/pip_preflight.sh <pip> <MB necessários> [dir de teste]
# Errno 122 (EDQUOT) acontece com o volume livre: a cota do usuário é outra.
set -u
PIP="${1:?pip}"; NEED_MB="${2:-6000}"; DIR="${3:-.}"

CACHE="$("$PIP" cache dir 2>/dev/null || echo "$HOME/.cache/pip")"
if [ -d "$CACHE" ]; then
    MB=$(du -sm "$CACHE" 2>/dev/null | cut -f1)
    echo "→ Limpando cache do pip (${MB:-0} MB em $CACHE)..."
    "$PIP" cache purge >/dev/null 2>&1 || rm -rf "$CACHE"
fi

FREE_MB=$(df -Pm "$DIR" | awk 'NR==2{print $4}')
if [ "${FREE_MB:-0}" -lt "$NEED_MB" ]; then
    echo "✗ Só ${FREE_MB} MB livres em $DIR (precisa de ~${NEED_MB} MB)." >&2
    exit 1
fi

# Teste real de escrita: pega cota de usuário/projeto, que o df não mostra.
PROBE="$DIR/.quota_probe.$$"
trap 'rm -f "$PROBE"' EXIT
if ! dd if=/dev/zero of="$PROBE" bs=1M count="$NEED_MB" status=none 2>/dev/null; then
    echo "✗ Cota de disco insuficiente: não foi possível gravar ${NEED_MB} MB em $DIR." >&2
    echo "  Libere espaço ou use 'make install-cpu' (bem menor, sem libs nvidia)." >&2
    exit 1
fi
echo "✓ Espaço e cota ok (${NEED_MB} MB gravados no teste)."
