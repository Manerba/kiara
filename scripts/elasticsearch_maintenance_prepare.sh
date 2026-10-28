#!/usr/bin/env bash
# S5 only: establish offline maintenance and save rollback inputs. Never deploys.
set -euo pipefail
if [[ $# != 3 || ${KIARA_ES_NO_OTHER_WRITERS:-} != yes ]]; then
    echo 'Usage: KIARA_ES_NO_OTHER_WRITERS=yes scripts/elasticsearch_maintenance_prepare.sh CONFIG EVIDENCE_DIR PYTHON' >&2
    echo 'First finish/stop all manual CLI, external ingestion and admin writers; see runbook.' >&2
    exit 2
fi
config=$(realpath "$1")
evidence=$(realpath -m "$2")
# Preserve the venv entry point: resolving its symlink selects system Python.
python=$(realpath -s "$3")
case "$evidence/" in
    /root/kiara/*|/opt/kiara/*|/etc/kiara/*) echo 'Evidence must be outside source/runtime/config' >&2; exit 2 ;;
esac
cd /root/kiara
umask 077
mkdir "$evidence"
# Fail before changing services and capture the actual interpreter environment.
"$python" - > "$evidence/python-environment.json" <<'PY'
import json
import sys
import elasticsearch
import psycopg2
import qdrant_client
import yaml
print(json.dumps({"executable": sys.executable, "prefix": sys.prefix,
                  "base_prefix": sys.base_prefix}, indent=2))
PY
"$python" -m pip freeze > "$evidence/python-packages.txt"
marker=$("$python" - "$config" <<'PY'
import sys
from kiara.config import load_config
from kiara.retrieval.elasticsearch_maintenance import marker_path
print(marker_path(load_config(sys.argv[1])))
PY
)
# Restrict syntax used in a systemd condition, without shell/systemd escaping ambiguity.
[[ "$marker" =~ ^/[a-zA-Z0-9_./-]+$ ]]
printf '%s\n' "$marker" > "$evidence/marker-path"
services=(kiara.service kiara-admin.service kiara-admin-chat.service kiara-worker.service kiara-mcp.service kiara-sync.service)
systemctl is-active "${services[@]}" kiara-sync.timer > "$evidence/services-before.txt" || true
# Install conditions before marker creation so any subsequent restart is blocked.
for service in "${services[@]}"; do
    install -d "/etc/systemd/system/$service.d"
    dropin="/etc/systemd/system/$service.d/69-elasticsearch-maintenance.conf"
    if [[ -e "$dropin" ]]; then
        echo "Existing maintenance condition: $dropin; inspect before another prepare" >&2
        exit 1
    fi
    printf '[Unit]\nConditionPathExists=!%s\n' "$marker" > "$dropin"
done
systemctl daemon-reload
"$python" -m kiara.retrieval.elasticsearch_maintenance begin --config "$config" --report "$evidence/begin.json"
if [[ $(systemctl show -p LoadState --value kiara-sync.timer) != not-found ]]; then
    systemctl stop kiara-sync.timer
fi
for service in "${services[@]}"; do
    if [[ $(systemctl show -p LoadState --value "$service") != not-found ]]; then systemctl stop "$service"; fi
done
"$python" - <<'PY'
from kiara.retrieval.elasticsearch_maintenance import assert_stopped
assert_stopped()
PY
# Read-only copies of runtime/code/config including dependencies. Root-only evidence.
cp -a --reflink=auto /opt/kiara "$evidence/runtime"
cp -a /etc/kiara "$evidence/config"
cp -a --reflink=auto /root/kiara "$evidence/source"
"$python" -m kiara.retrieval.elasticsearch_maintenance retained --config "$config" --report "$evidence/retained-before.json"
echo "Offline maintenance established; rollback inputs: $evidence. Continue with the runbook."
