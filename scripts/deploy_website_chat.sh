#!/usr/bin/env bash
# Aufruf ausschliesslich ueber ./deploy.sh --website-chat.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="root@update.example.com"
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10)
BACKUP_PATH="/var/backups/kiara-chat/providers-$(date -u +%Y%m%dT%H%M%S)"

# Install only the additional provider dependencies before stopping services.
ssh "${SSH_OPTIONS[@]}" "$SERVER" '/opt/kiara-chat/venv/bin/pip install --disable-pip-version-check -r /dev/stdin' < "$PROJECT_DIR/website/chat_backend/requirements.txt"
ssh "${SSH_OPTIONS[@]}" "$SERVER" "bash -se -- '$BACKUP_PATH'" <<'REMOTE'
umask 077
BACKUP_PATH="$1"
install -d -m 0700 "$BACKUP_PATH" /etc/kiara-chat /var/lib/kiara-chat
cp -a /opt/kiara-chat/admin "$BACKUP_PATH/"
for name in app.py chat_settings.py llm_providers.py migrate_settings.py config.yaml settings.json knowledge.md admin.py; do
    if [ -f "/opt/kiara-chat/$name" ]; then cp -p "/opt/kiara-chat/$name" "$BACKUP_PATH/"; fi
done
cp -p /etc/apache2/sites-available/003-kiara-ai-le-ssl.conf "$BACKUP_PATH/"
cp -Lp /etc/apache2/sites-enabled/003-kiara-ai-le-ssl.conf "$BACKUP_PATH/003-kiara-ai-le-ssl.enabled.conf"
systemctl stop kiara-chat-admin.service kiara-chat.service
/opt/kiara-chat/venv/bin/python - "$BACKUP_PATH" <<'PY'
import sqlite3, sys
from pathlib import Path
for path in [Path('/opt/kiara-chat/chat.db'), Path('/var/lib/kiara-chat/settings.sqlite')]:
    if path.exists():
        with sqlite3.connect(path) as source, sqlite3.connect(Path(sys.argv[1])/path.name) as target:
            source.backup(target)
PY
echo "Website-Chat-Sicherung: $BACKUP_PATH"
REMOTE

# No --delete here: the destination also contains live data, venv and dashboard.
rsync -az --exclude '__pycache__/' --exclude 'tests/' \
    "$PROJECT_DIR/website/chat_backend/" "$SERVER:/opt/kiara-chat/"

ssh "${SSH_OPTIONS[@]}" "$SERVER" "bash -se -- '$BACKUP_PATH'" <<'REMOTE'
umask 077
BACKUP_PATH="$1"
/opt/kiara-chat/venv/bin/python /opt/kiara-chat/migrate_settings.py
# The runtime now uses the atomic settings store, including encrypted keys.
for name in config.yaml settings.json knowledge.md admin.py; do
    if [ -f "/opt/kiara-chat/$name" ]; then mv "/opt/kiara-chat/$name" "$BACKUP_PATH/"; fi
done
install -d -m 0755 /etc/systemd/system/kiara-chat.service.d
cat > /etc/systemd/system/kiara-chat.service.d/provider.conf <<'UNIT'
[Service]
ExecStart=
ExecStart=/opt/kiara-chat/venv/bin/uvicorn app:app --host 127.0.0.1 --port 8600 --no-proxy-headers
UMask=0077
UNIT
chmod 0644 /etc/systemd/system/kiara-chat.service.d/provider.conf

python3 - <<'PY'
from pathlib import Path
paths = {Path('/etc/apache2') / directory / '003-kiara-ai-le-ssl.conf'
         for directory in ('sites-available', 'sites-enabled')}
# Some installations keep a regular copy in sites-enabled instead of a symlink.
for p in {path.resolve(strict=True) for path in paths}:
    text = p.read_text()
    marker = '    # KIara chat request identity\n'
    if marker not in text:
        text = text.replace('    # Chat API reverse proxy\n', marker + '''    <Location /api/>
        RequestHeader unset Forwarded
        RequestHeader unset X-Forwarded-For
        RequestHeader unset CF-Connecting-IP
        RequestHeader set X-Kiara-Chat-Peer "expr=%{CONN_REMOTE_ADDR}"
        LimitRequestBody 100000
    </Location>
    # Chat API reverse proxy
''')
        if marker not in text:
            raise RuntimeError('Chat proxy configuration not found')
    text = text.replace('ProxyPass /api/ http://127.0.0.1:8600/api/\n', 'ProxyPass /api/ http://127.0.0.1:8600/api/ connectiontimeout=5 timeout=210\n')
    p.write_text(text)
PY
cat > /etc/apache2/conf-available/kiara-chat-backend-private.conf <<'APACHE'
<Directory "/var/www/kiara-ai/chat_backend">
    AllowOverride None
    Require all denied
</Directory>
APACHE
chmod 0644 /etc/apache2/conf-available/kiara-chat-backend-private.conf
a2enconf kiara-chat-backend-private
apache2ctl configtest
systemctl daemon-reload
systemctl start kiara-chat.service
/opt/kiara-chat/venv/bin/python - <<'PY'
import time, urllib.request
for attempt in range(30):
    try:
        with urllib.request.urlopen('http://127.0.0.1:8600/api/config', timeout=2) as response:
            assert response.status == 200
        break
    except Exception:
        if attempt == 29:
            raise
        time.sleep(0.25)
PY
systemctl reload apache2.service
systemctl is-active kiara-chat.service apache2.service
REMOTE
