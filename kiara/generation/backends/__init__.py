"""LLM-Backend-Abstraction — einheitliches Interface fuer alle LLM-Provider.

Stellt das LLMBackend-ABC, drei Provider-Implementierungen (Ollama, Anthropic,
OpenAI), einen Anonymisierungs-Decorator und eine Factory-Funktion bereit.
"""

from kiara.generation.backends.anonymizing import AnonymizingBackend
from kiara.generation.backends.anthropic import AnthropicBackend
from kiara.generation.backends.base import LLMBackend
from kiara.generation.backends.factory import create_backend, wrap_with_anonymization
from kiara.generation.backends.ollama import OllamaBackend
from kiara.generation.backends.openai import OpenAIBackend
