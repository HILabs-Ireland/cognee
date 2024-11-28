import logging
from typing import List, Optional
import litellm
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine

litellm.set_verbose = False
logger = logging.getLogger("LiteLLMEmbeddingEngine")

class LiteLLMEmbeddingEngine(EmbeddingEngine):
    api_key: str
    endpoint: str
    api_version: str
    model: str
    dimensions: int

    def __init__(
        self,
        model: Optional[str] = "text-embedding-3-large",
        dimensions: Optional[int] = 3072,
        api_key: str = None,
        endpoint: str = None,
        api_version: str = None,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.api_version = api_version
        self.model = model
        self.dimensions = dimensions

    async def embed_text(self, text: List[str]) -> List[List[float]]:
        async def get_embedding(text_):
            try:
                response = await litellm.aembedding(
                    self.model,
                    input = text_,
                    api_key = self.api_key,
                    api_base = self.endpoint,
                    api_version = self.api_version
                )
            except litellm.exceptions.BadRequestError as error:
                logger.error("Error embedding text: %s", str(error))
                raise error

            return [data["embedding"] for data in response.data]

        # tasks = [get_embedding(text_) for text_ in text]
        result = await get_embedding(text)
        return result

    def get_vector_size(self) -> int:
        return self.dimensions
