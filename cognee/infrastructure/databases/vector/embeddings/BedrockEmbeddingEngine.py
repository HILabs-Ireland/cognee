import boto3
import json
import numpy as np
from typing import List, Optional
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import EmbeddingEngine
from cognee.infrastructure.databases.exceptions.EmbeddingException import EmbeddingException
from cognee.infrastructure.llm.embedding_rate_limiter import (
    embedding_rate_limit_async,
    embedding_sleep_and_retry_async,
)

logger = get_logger("BedrockEmbeddingEngine")

class BedrockEmbeddingEngine(EmbeddingEngine):
    def __init__(
        self,
        model: str = "amazon.titan-embed-text-v2:0",
        dimensions: int = 1024,
        max_tokens: int = 8192,
        aws_access_key_id: str = None,
        aws_secret_access_key: str = None,
        aws_region: str = None,
        aws_session_token: str = None,
    ):
        import os
        
        # Initialize AWS credentials
        aws_config = {
            'service_name': 'bedrock-runtime'
        }
        
        if aws_access_key_id and aws_secret_access_key:
            aws_config.update({
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key,
            })
            if aws_session_token:
                aws_config['aws_session_token'] = aws_session_token
        elif os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
            aws_config.update({
                'aws_access_key_id': os.environ['AWS_ACCESS_KEY_ID'],
                'aws_secret_access_key': os.environ['AWS_SECRET_ACCESS_KEY'],
            })
            if os.environ.get('AWS_SESSION_TOKEN'):
                aws_config['aws_session_token'] = os.environ['AWS_SESSION_TOKEN']
        
        if aws_region:
            aws_config['region_name'] = aws_region
        elif os.environ.get('AWS_REGION'):
            aws_config['region_name'] = os.environ['AWS_REGION']

        self.bedrock = boto3.client(**aws_config)
        self.model = model
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.tokenizer = self.get_tokenizer()

    def get_tokenizer(self):
        """Get the appropriate tokenizer for the model."""
        from cognee.infrastructure.llm.tokenizer.HuggingFace import HuggingFaceTokenizer
        return HuggingFaceTokenizer(model="bert-base-uncased", max_tokens=self.max_tokens)

    @embedding_sleep_and_retry_async()
    @embedding_rate_limit_async
    async def aembed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        try:
            logger.info(f"Generating embeddings for text: {text[:100]}...")
            logger.info(f"Using Bedrock model: {self.model}")
            
            if 'amazon.titan-embed' in self.model:
                body = {
                    "inputText": text
                }
                logger.debug(f"Request body: {json.dumps(body)}")
            else:
                raise EmbeddingException(f"Unsupported model: {self.model}")
            
            try:
                response = self.bedrock.invoke_model(
                    modelId=self.model,
                    body=json.dumps(body)
                )
                logger.debug(f"Bedrock raw response: {response}")
            except Exception as e:
                logger.error(f"Bedrock API error: {str(e)}")
                if hasattr(e, 'response'):
                    logger.error(f"Response status: {e.response.get('ResponseMetadata', {}).get('HTTPStatusCode')}")
                    logger.error(f"Response body: {e.response.get('Error', {})}")
                raise
            
            response_body = json.loads(response['body'].read())
            logger.debug(f"Parsed response body: {response_body}")
            
            embeddings = response_body.get('embedding')
            if not embeddings:
                raise EmbeddingException(f"No embeddings found in response from model {self.model}")
            
            return embeddings

        except Exception as e:
            logger.error(f"Error getting embeddings from Bedrock: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            if hasattr(e, '__dict__'):
                logger.error(f"Error details: {e.__dict__}")
            raise EmbeddingException(f"Failed to get embeddings: {str(e)}")

    @embedding_sleep_and_retry_async()
    @embedding_rate_limit_async
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents."""
        embeddings = []
        for text in texts:
            embedding = await self.aembed_query(text)
            embeddings.append(embedding)
        return embeddings

    async def embed_text(self, text: str) -> List[List[float]]:
        """Embed a single text string."""
        if isinstance(text, str):
            text = [text]
        return await self.aembed_documents(text)

    def get_vector_size(self) -> int:
        """Return the size of the embedding vectors."""
        return self.dimensions
