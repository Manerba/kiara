"""Legacy-Shim: leitet auf kiara_document_loaders.vision um."""

import sys

from kiara_document_loaders import vision as _impl

sys.modules[__name__] = _impl
