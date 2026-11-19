# Re-Exports fuer Rueckwaertskompatibilitaet
from kiara_ingestion_core.datasource_runtime import (
    build_datasource_overrides as _build_datasource_overrides,
)
from kiara_ingestion_core.datasource_runtime import (
    load_connector_config as load_connector_config,
)


def build_datasource_overrides(row):
    overrides = _build_datasource_overrides(row)
    for target in ("vector", "relational"):
        key = f"ds_cad_{target}_enabled"
        try:
            value = row[key]
        except (KeyError, IndexError):
            value = True
        overrides[f"cad_{target}_enabled"] = value is not False
    return overrides
