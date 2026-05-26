"""Legacy-Shim: leitet auf kiara_document_loaders.general um."""

import sys

from kiara_document_loaders import general as _impl

sys.modules[__name__] = _impl
