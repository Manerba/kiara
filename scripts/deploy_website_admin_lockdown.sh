#!/usr/bin/env bash
# Sofortsperre: Dashboard stoppen und den HTTPS-Zugang entfernen.
set -euo pipefail
ssh -o BatchMode=yes -o ConnectTimeout=10 root@update.example.com 'bash -se' <<'REMOTE'
umask 077
systemctl stop kiara-chat-admin.service
if [ -e /etc/apache2/sites-enabled/007-kiara-dashboard-ssl.conf ]; then
    a2dissite 007-kiara-dashboard-ssl.conf
fi
install -d -m 0700 /var/backups/kiara-chat
if [ -e /var/www/kiara-ai/admin ]; then
    BACKUP_DIR="$(mktemp -d /var/backups/kiara-chat/admin-lockdown-XXXXXXXX)"
    mv /var/www/kiara-ai/admin "$BACKUP_DIR/public-admin"
fi
cat > /etc/apache2/conf-available/kiara-chat-admin-private.conf <<'APACHE'
<Directory "/var/www/kiara-ai/admin">
    AllowOverride None
    Require all denied
</Directory>
APACHE
chmod 0644 /etc/apache2/conf-available/kiara-chat-admin-private.conf
a2enconf kiara-chat-admin-private
apache2ctl configtest
systemctl reload apache2.service
systemctl is-active apache2.service kiara-chat.service
echo "Website-Dashboard gestoppt; HTTPS-Zugang gesperrt."
REMOTE
