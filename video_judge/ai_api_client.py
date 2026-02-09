"""AI SDK client wrappers with lazy initialization."""

import os
from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic
from dotenv import load_dotenv

from google import genai
from openai import OpenAI
import anthropic

T = TypeVar('T')


class AIAPIClientBase(ABC, Generic[T]):
    """Base class for lazy-loaded SDK clients.

    Provides:
    - Lazy initialization (dotenv only loads when first accessed)
    - Type safety via Generic[T]
    - Reset capability for testing/mocking

    Usage:
        class MyClient(AIAPIClientBase[SomeSDKClient]):
            def _initialize(self) -> SomeSDKClient:
                return SomeSDKClient(api_key=os.getenv("API_KEY"))

        my_client = MyClient()
        my_client.client.do_something()  # Initializes on first access
    """

    def __init__(self):
        self._client: Optional[T] = None
        self._initialized = False

    @abstractmethod
    def _initialize(self) -> T:
        """Override to create the actual SDK client.

        This method is called once on first access to .client property.
        Load environment variables and instantiate your SDK client here.
        """
        pass

    @property
    def client(self) -> T:
        """Lazy-load and return the client.

        First access triggers:
        1. dotenv load
        2. _initialize() call
        3. Caching of result

        Subsequent accesses return cached client.
        """
        if not self._initialized:
            load_dotenv()  # Only load once when first client needs it
            self._client = self._initialize()
            self._initialized = True
        return self._client

    def reset(self):
        """Reset client state.

        Useful for testing - allows re-initialization with different
        environment variables or mocked dependencies.
        """
        self._client = None
        self._initialized = False


class GeminiAPIClient(AIAPIClientBase[genai.Client]):
    """Lazy-loaded Google Gemini API client.

    Requires GEMINI_API_KEY in environment.
    """

    def _initialize(self) -> genai.Client:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        return genai.Client(api_key=api_key)


class OpenAIAPIClient(AIAPIClientBase[OpenAI]):
    """Lazy-loaded OpenAI API client.

    Requires OPENAI_API_KEY in environment.
    """

    def _initialize(self) -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return OpenAI(api_key=api_key)


class AnthropicAPIClient(AIAPIClientBase[anthropic.Anthropic]):
    """Lazy-loaded Anthropic API client.

    Anthropic SDK automatically reads ANTHROPIC_API_KEY from environment.
    """

    def _initialize(self) -> anthropic.Anthropic:
        return anthropic.Anthropic()


google_client = GeminiAPIClient()
openai_client = OpenAIAPIClient()
anthropic_client = AnthropicAPIClient()
