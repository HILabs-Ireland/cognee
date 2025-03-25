import asyncio
import aiohttp
import logging
from typing import List, Optional
import os

import aiohttp.http_exceptions

from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.exceptions.EmbeddingException import EmbeddingException
from cognee.infrastructure.llm.tokenizer.HuggingFace import HuggingFaceTokenizer

logger = logging.getLogger("OllamaEmbeddingEngine")


class OllamaEmbeddingEngine(EmbeddingEngine):
    model: str
    dimensions: int
    max_tokens: int
    endpoint: str
    mock: bool
    huggingface_tokenizer_name: str

    MAX_RETRIES = 5

    def __init__(
        self,
        model: Optional[str] = "avr/sfr-embedding-mistral:latest",
        dimensions: Optional[int] = 1024,
        max_tokens: int = 512,
        endpoint: Optional[str] = "http://localhost:11434/api/embeddings",
        huggingface_tokenizer: str = "Salesforce/SFR-Embedding-Mistral",
    ):
        self.model = model
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.endpoint = endpoint
        self.huggingface_tokenizer_name = huggingface_tokenizer
        self.tokenizer = self.get_tokenizer()

        enable_mocking = os.getenv("MOCK_EMBEDDING", "false")
        if isinstance(enable_mocking, bool):
            enable_mocking = str(enable_mocking).lower()
        self.mock = enable_mocking in ("true", "1", "yes")

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        """
        Given a list of text prompts, returns a list of embedding vectors.
        """
        if self.mock:
            return [[0.0] * self.dimensions for _ in text]

        embeddings = await asyncio.gather(*[self._get_embedding(prompt) for prompt in text])
        return embeddings

    async def _get_embedding(self, prompt: str) -> List[float]:
        """
        Internal method to call the Ollama embeddings endpoint for a single prompt.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
        }
        headers = {}
        api_key = os.getenv("LLM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        retries = 0
        while retries < self.MAX_RETRIES:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.endpoint, json=payload, headers=headers, timeout=60.0
                    ) as response:
                        data = await response.json()
                        return data["embedding"]
            except aiohttp.http_exceptions.HttpBadRequest as e:
                logger.error(f"HTTP error on attempt {retries + 1}: {e}")
                retries += 1
                await asyncio.sleep(min(2**retries, 60))
            except Exception as e:
                logger.error(f"Error on attempt {retries + 1}: {e}")
                retries += 1
                await asyncio.sleep(min(2**retries, 60))
        raise EmbeddingException(
            f"Failed to embed text using model {self.model} after {self.MAX_RETRIES} retries"
        )

    def get_vector_size(self) -> int:
        return self.dimensions

    def get_tokenizer(self):
        logger.debug("Loading HuggingfaceTokenizer for OllamaEmbeddingEngine...")
        tokenizer = HuggingFaceTokenizer(
            model=self.huggingface_tokenizer_name, max_tokens=self.max_tokens
        )
        logger.debug("Tokenizer loaded for OllamaEmbeddingEngine")
        return tokenizer
