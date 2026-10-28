"""Legacy-Shim: leitet auf kiara_document_loaders.content_detect um."""

import sys

from kiara_document_loaders import content_detect as _impl

sys.modules[__name__] = _impl
