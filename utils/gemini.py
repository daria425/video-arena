from typing import Optional, TypeVar, Type, List
from pydantic import BaseModel
from tenacity import (
    retry, wait_fixed, retry_if_not_exception_type, stop_after_attempt
)
from google.auth.exceptions import GoogleAuthError
from google.genai.errors import ServerError, ClientError
from config.genai_client import client as google_client
from config.logger import logger
from utils.file_utils import create_image_input
from google.genai import types

T = TypeVar("T", bound=BaseModel)


@retry(
    retry=retry_if_not_exception_type(
        [GoogleAuthError, ClientError, ServerError]),
    wait=wait_fixed(3),
    stop=stop_after_attempt(3)

)
def _call_gemini_with_image_list(
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

    contents = types.Content(role="user", parts=parts)  # Use parts!

    response = google_client.models.generate_content(
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
