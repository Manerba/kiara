#!/usr/bin/env bash
# Aufruf ueber ./deploy.sh --website-admin-https.
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER="root@update.example.com"
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10)

ssh "${SSH_OPTIONS[@]}" "$SERVER" 'bash -se' <<'REMOTE'
install -d -m 0755 /var/www/kiara-dashboard-acme
cat > /etc/apache2/sites-available/007-kiara-dashboard.conf <<'APACHE'
<VirtualHost *:80>
    ServerName dashboard.kiara-ai.de
    DocumentRoot /var/www/kiara-dashboard-acme
    <Directory /var/www/kiara-dashboard-acme>
        Options -Indexes
        AllowOverride None
        Require all granted
    </Directory>
    RewriteEngine On
    RewriteCond %{REQUEST_URI} !^/\.well-known/acme-challenge/
    RewriteRule ^ https://dashboard.kiara-ai.de%{REQUEST_URI} [R=308,L,NE]
</VirtualHost>
APACHE
a2ensite 007-kiara-dashboard.conf
apache2ctl configtest
systemctl reload apache2.service
certbot certonly --webroot -w /var/www/kiara-dashboard-acme \
    --cert-name dashboard.kiara-ai.de -d dashboard.kiara-ai.de \
    --non-interactive --agree-tos --keep-until-expiring

umask 077
install -d -m 0700 /var/backups/kiara-chat
BACKUP_DIR="$(mktemp -d /var/backups/kiara-chat/admin-https-XXXXXXXX)"
cp -a /opt/kiara-chat/admin "$BACKUP_DIR/"
cp -p /etc/kiara-chat/admin.env "$BACKUP_DIR/"
systemctl stop kiara-chat-admin.service
echo "Dashboard-Sicherung: $BACKUP_DIR"
REMOTE

rsync -az --delete --exclude '__pycache__/' --exclude 'tests/' --exclude '*.sh' \
    "$PROJECT_DIR/website/admin/" "$SERVER:/opt/kiara-chat/admin/"

ssh "${SSH_OPTIONS[@]}" "$SERVER" 'bash -se' <<'REMOTE'
umask 077
install -d -m 0700 /var/lib/kiara-chat /etc/kiara-chat
cd /opt/kiara-chat/admin
/opt/kiara-chat/venv/bin/python - <<'PY'
import json
import os
from pathlib import Path
from dashboard_auth import hash_password

auth = Path("/etc/kiara-chat/admin-auth.json")
env = Path("/etc/kiara-chat/admin.env")
if not auth.exists():
    old = dict(line.split("=", 1) for line in env.read_text().splitlines() if "=" in line)
    password = old["KIARA_CHAT_ADMIN_PASSWORD"]
    if len(password) < 32:
        raise RuntimeError("Vor der Freigabe muss ein starkes Admin-Passwort gesetzt sein")
    initial = Path("/etc/kiara-chat/admin-initial-password")
    initial.write_text(password + "\n")
    initial.chmod(0o600)
    auth.write_text(json.dumps({
        "username": old.get("KIARA_CHAT_ADMIN_USER", "admin"),
        "password_hash": hash_password(password),
        "public_origin": "https://dashboard.kiara-ai.de",
        "session_db": "/var/lib/kiara-chat/admin-auth.sqlite",
        "session_seconds": 28800,
        "idle_seconds": 1800,
        "login_window_seconds": 600,
        "login_attempts_per_ip": 5,
        "login_attempts_global": 50,
    }, indent=2) + "\n")
    auth.chmod(0o600)
env.write_text("KIARA_CHAT_ADMIN_AUTH_FILE=/etc/kiara-chat/admin-auth.json\n")
env.chmod(0o600)
PY
cat > /etc/systemd/system/kiara-chat-admin.service.d/private.conf <<'UNIT'
[Service]
EnvironmentFile=/etc/kiara-chat/admin.env
Environment=PYTHONPATH=/opt/kiara-chat
ExecStart=
ExecStart=/opt/kiara-chat/venv/bin/uvicorn admin:app --host 127.0.0.1 --port 8601 --no-proxy-headers
UMask=0077
UNIT
chmod 0644 /etc/systemd/system/kiara-chat-admin.service.d/private.conf
systemctl daemon-reload
systemctl start kiara-chat-admin.service

# Erst nach erfolgreichem Start der neuen Authentifizierung oeffentlich routen.
/opt/kiara-chat/venv/bin/python - <<'PY'
import time
import urllib.request
for attempt in range(20):
    try:
        req = urllib.request.Request("http://127.0.0.1:8601/login", headers={
            "Host": "dashboard.kiara-ai.de", "X-Forwarded-Proto": "https"})
        with urllib.request.urlopen(req, timeout=2) as response:
            assert response.status == 200 and b'name="password"' in response.read()
        break
    except Exception:
        if attempt == 19:
            raise
        time.sleep(0.25)
PY

cat > /etc/apache2/sites-available/007-kiara-dashboard-ssl.conf <<'APACHE'
<VirtualHost *:443>
    ServerName dashboard.kiara-ai.de
    Include /etc/letsencrypt/options-ssl-apache.conf
    SSLCertificateFile /etc/letsencrypt/live/dashboard.kiara-ai.de/fullchain.pem
    SSLCertificateKeyFile /etc/letsencrypt/live/dashboard.kiara-ai.de/privkey.pem
    ProxyRequests Off
    ProxyPreserveHost On
    ProxyAddHeaders Off
    RequestHeader unset Forwarded
    RequestHeader unset X-Forwarded-For
    RequestHeader unset CF-Connecting-IP
    RequestHeader set X-Forwarded-Proto "https"
    RequestHeader set X-Kiara-Admin-Peer "expr=%{CONN_REMOTE_ADDR}"
    ProxyPass / http://127.0.0.1:8601/ connectiontimeout=5 timeout=30
    ProxyPassReverse / http://127.0.0.1:8601/
    LimitRequestBody 1048576
    Header always set Strict-Transport-Security "max-age=31536000"
    Header always set X-Content-Type-Options "nosniff"
    Header always set X-Frame-Options "DENY"
    Header always set Referrer-Policy "same-origin"
    Header always set Permissions-Policy "camera=(), microphone=(), geolocation=()"
    ErrorLog ${APACHE_LOG_DIR}/kiara-dashboard-error.log
    CustomLog ${APACHE_LOG_DIR}/kiara-dashboard-access.log combined
</VirtualHost>
APACHE
chmod 0644 /etc/apache2/sites-available/007-kiara-dashboard*.conf
a2ensite 007-kiara-dashboard-ssl.conf
apache2ctl configtest
systemctl reload apache2.service

install -d -m 0755 /etc/letsencrypt/renewal-hooks/deploy
cat > /etc/letsencrypt/renewal-hooks/deploy/kiara-dashboard-apache <<'HOOK'
#!/bin/sh
set -e
apache2ctl configtest
systemctl reload apache2.service
HOOK
chmod 0755 /etc/letsencrypt/renewal-hooks/deploy/kiara-dashboard-apache
systemctl is-active apache2.service kiara-chat.service kiara-chat-admin.service certbot.timer
REMOTE
