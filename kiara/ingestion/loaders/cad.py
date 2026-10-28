"""Legacy-Shim: leitet auf kiara_document_loaders.cad um."""

import sys

from kiara_document_loaders import cad as _impl

sys.modules[__name__] = _impl
