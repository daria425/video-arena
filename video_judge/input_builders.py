from typing import Optional, TypeVar, Type, List
from pydantic import BaseModel
from tenacity import (
    retry, wait_fixed, retry_if_not_exception_type, stop_after_attempt
)
from google.auth.exceptions import GoogleAuthError
from google.genai.errors import ServerError, ClientError
from video_judge.ai_api_client import google_client, openai_client, anthropic_client
from video_judge.config.logger import logger
from video_judge.utils.file_utils import create_image_input
from google.genai import types
from openai import AuthenticationError, RateLimitError, PermissionDeniedError
import base64

T = TypeVar("T", bound=BaseModel)


@retry(
    retry=retry_if_not_exception_type(
        (GoogleAuthError, ClientError, ServerError)),
    wait=wait_fixed(3),
    stop=stop_after_attempt(3)

)
def build_gemini_input_with_image_list(
    *,
    image_bytes_list: List[bytes],
    user_prompt_list: List[str],
    system_instruction: str,
    response_schema: Optional[Type[T]] = None,
    model: str = "gemini-2.5-pro",
):
    parts = []

    for image_bytes, user_prompt in zip(image_bytes_list, user_prompt_list):
        image_input = create_image_input(image_bytes)
        user_input = types.Part.from_text(text=user_prompt)
        parts.extend([image_input, user_input])

    if len(user_prompt_list) > len(image_bytes_list):
        for extra_prompt in user_prompt_list[len(image_bytes_list):]:
            parts.append(types.Part.from_text(text=extra_prompt))

    generation_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0
    )
    if response_schema:
        generation_config.response_schema = response_schema
        generation_config.response_mime_type = "application/json"

    contents = types.Content(role="user", parts=parts)

    response = google_client.client.models.generate_content(
        model=model,
        contents=contents,
        config=generation_config,
    )
    logger.debug(f"Recieved response: {response}")
    if response_schema:
        parsed = response.parsed
        if not parsed:
            raise ValueError(
                f"Gemini returned empty/invalid response for schema {response_schema.__name__}")
        return parsed
    return response.text


@retry(
    retry=retry_if_not_exception_type(
        (AuthenticationError, RateLimitError, PermissionDeniedError)),
    wait=wait_fixed(3),
    stop=stop_after_attempt(3)

)
def build_openai_input_with_image_list(
    *,
    image_bytes_list: List[bytes],
    user_prompt_list: List[str],
    system_instruction: str,
    response_schema: Optional[Type[T]] = None,
    model: str = "gpt-4o",
):
    input_list = [
        {"role": "user", "content": []}
    ]
    for image_bytes, user_prompt in zip(image_bytes_list, user_prompt_list):
        b64_str = base64.b64encode(image_bytes).decode("utf-8")
        image_input = {
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64_str}",
        }
        text_input = {
            "type": "input_text",
            "text": user_prompt
        }
        input_list[0]["content"].extend([image_input, text_input])
    if len(user_prompt_list) > len(image_bytes_list):
        for extra_prompt in user_prompt_list[len(image_bytes_list):]:
            input_list[0]["content"].append({
                "type": "input_text",
                "text": extra_prompt
            })
    if response_schema:
        response = openai_client.client.responses.parse(model=model, temperature=0,
                                                        text_format=response_schema, input=input_list, instructions=system_instruction)
        parsed = response.output_parsed
        if not parsed:
            raise ValueError(
                f"OpenAI returned empty/invalid response for schema {response_schema.__name__}")
        return parsed
    else:
        response = openai_client.client.responses.create(
            model=model, input=input_list, instructions=system_instruction, temperature=0
        )
        return response.output_text


def build_claude_input_with_image_list(
    *,
    image_bytes_list: List[bytes],
    user_prompt_list: List[str],
    system_instruction: str,
    response_schema: Optional[Type[T]] = None,
    model: str = "claude-sonnet-3-5",
):
    input_list = [
        {"role": "user", "content": []}
    ]
    for image_bytes, user_prompt in zip(image_bytes_list, user_prompt_list):
        b64_str = base64.b64encode(image_bytes).decode("utf-8")
        image_input = {
            "type": "image",
            "source": {
                "data": f"data:image/jpeg;base64,{b64_str}",
                "media_type": "image/jpeg",
                "type": "base64"
            }
        }
        text_input = {
            "type": "text",
            "text": user_prompt
        }
        input_list[0]["content"].extend([image_input, text_input])
    if len(user_prompt_list) > len(image_bytes_list):
        for extra_prompt in user_prompt_list[len(image_bytes_list):]:
            input_list[0]["content"].append({
                "type": "text",
                "text": extra_prompt
            })
    if response_schema:
        response = anthropic_client.client.messages.parse(
            messages=input_list,
            model=model,
            output_format=response_schema,
            temperature=0,
            system=system_instruction

        )
        parsed = response.parsed_output
        if not parsed:
            raise ValueError(
                f"Claude returned empty/invalid response for schema {response_schema.__name__}")
        return parsed
    else:
        response = anthropic_client.client.messages.create(
            messages=input_list,
            model=model,
            temperature=0,
            system=system_instruction
        )
        return response
