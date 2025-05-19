import os
import boto3
import json
from typing import Type, Optional
from pydantic import BaseModel
import instructor
from cognee.shared.logging_utils import get_logger

from cognee.exceptions import InvalidValueError
from cognee.infrastructure.llm.llm_interface import LLMInterface
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.infrastructure.llm.rate_limiter import rate_limit_async, sleep_and_retry_async


class BedrockAdapter(LLMInterface):
    """Adapter for Amazon Bedrock API, focusing on Claude models"""

    name = "Bedrock"
    model: str
    logger = get_logger("BedrockAdapter")

    def __init__(
        self,
        max_tokens: int,
        model: str = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = None,
        aws_session_token: Optional[str] = None
    ):
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
        else:
            aws_config['region_name'] = 'us-east-1'

        self.bedrock = boto3.client(**aws_config)
        
        self.client = instructor.from_bedrock(self.bedrock)
        
        self.model = model or "anthropic.claude-3-haiku-20240307-v1:0"
        self.max_tokens = max_tokens
        
    @sleep_and_retry_async()
    @rate_limit_async
    async def acreate_structured_output(
        self, text_input: str, system_prompt: str, response_model: Type[BaseModel]
    ) -> BaseModel:
        """Generate a response from a user query using Amazon Bedrock."""
        self.logger.info(f"Using model: {self.model}")
        
        try:
            try:
                # Use instructor's chat completions API
                response = self.client.chat.completions.create(
                    modelId=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [{
                                "text": f"Use the given format to extract information from the following input: {text_input}. {system_prompt}"
                            }]
                        }
                    ],
                    response_model=response_model
                )
                self.logger.debug(f"Structured response: {response}")
                return response
            except Exception as e:
                self.logger.error(f"Error creating structured output: {str(e)}")
                raise InvalidValueError(message=f"Error invoking Bedrock model: {str(e)}")
        except Exception as e:
            raise InvalidValueError(message=f"Error invoking Bedrock model: {str(e)}")

    def show_prompt(self, text_input: str, system_prompt: str) -> str:
        """Format and display the prompt for a user query."""

        if not text_input:
            text_input = "No user input provided."
        if not system_prompt:
            raise InvalidValueError(message="No system prompt path provided.")

        system_prompt = read_query_prompt(system_prompt)

        formatted_prompt = (
            f"""System Prompt:\n{system_prompt}\n\nUser Input:\n{text_input}\n"""
            if system_prompt
            else None
        )

        return formatted_prompt
