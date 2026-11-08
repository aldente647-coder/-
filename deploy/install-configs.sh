#!/bin/bash
# Разложить конфиги Asterisk на машине.
#
# Главное здесь — что подставляется, а что нет.
#
# В extensions.conf переменные вида ${STASIS_APP} и ${AMDSTATUS} — это
# переменные ДИЛПЛАНА, которые раскрывает сам Asterisk во время звонка.
# Прогнать этот файл через envsubst значит заменить их пустотой ещё
# при установке. Дилплан останется синтаксически верным, Asterisk
# запустится без единой жалобы, а в логе будет `Stasis(,,)` — канал
# не войдёт ни в какое приложение, и звонок оборвётся молча.
#
# Поэтому подстановка делается только там, где лежат секреты, и только
# для явно перечисленных переменных.
set -euo pipefail

SRC="$(cd "$(dirname "$0")/asterisk" && pwd)"
DST="${1:-/etc/asterisk}"
: "${SOCMON_ARI_PASSWORD:?не задана SOCMON_ARI_PASSWORD}"

# Копируются как есть: секретов нет, переменные — дилплана и Asterisk.
for f in extensions.conf amd.conf rtp.conf modules.conf manager.conf http.conf; do
  install -m 640 -o root -g asterisk "$SRC/$f" "$DST/$f"
done

# Подстановка строго по списку переменных: всё, что не перечислено,
# остаётся нетронутым.
envsubst '$SOCMON_ARI_PASSWORD' < "$SRC/ari.conf" > "$DST/ari.conf"
chown root:asterisk "$DST/ari.conf"; chmod 640 "$DST/ari.conf"

if [ -f "$SRC/pjsip.conf" ] && [ "${SOCMON_SKIP_PJSIP:-}" != "1" ]; then
  envsubst '$SOCMON_EXTERNAL_IP $SOCMON_TRUNK_HOST $SOCMON_TRUNK_USER
            $SOCMON_TRUNK_PASSWORD $SOCMON_OPERATOR1_PASSWORD' \
    < "$SRC/pjsip.conf" > "$DST/pjsip.conf"
  chown root:asterisk "$DST/pjsip.conf"; chmod 640 "$DST/pjsip.conf"
fi

# Проверка: переменные дилплана обязаны остаться на месте.
for var in STASIS_APP SOCMON_ATTEMPT AMDSTATUS; do
  grep -q "\${$var}" "$DST/extensions.conf" \
    || { echo "ОШИБКА: \${$var} затёрт в extensions.conf" >&2; exit 1; }
done
echo "конфиги разложены в $DST, переменные дилплана целы"
