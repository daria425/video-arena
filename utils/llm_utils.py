from typing import Optional, TypeVar, Type, List
from pydantic import BaseModel
from config.genai_client import client as google_client
from config.logger import logger
from utils.format import format_prompt
from utils.file_utils import create_image_input
from google.genai import types

T = TypeVar("T", bound=BaseModel)


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
        return response.parsed
    return response.text


def _call_gemini_with_image(
    *,
    image_bytes: bytes,
    user_prompt: str,
    system_prompt_path: str,
    response_schema: Optional[Type[T]] = None,
    model: str = "gemini-2.5-pro",
):
    system_instruction = format_prompt(system_prompt_path)
    image_input = create_image_input(image_bytes)
    user_input = types.Part.from_text(text=user_prompt)

    generation_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
    )
    if response_schema:
        generation_config.response_schema = response_schema
        generation_config.response_mime_type = "application/json"
    contents = types.Content(role="user", parts=[image_input, user_input])
    response = google_client.models.generate_content(
        model=model,
        contents=contents,
        config=generation_config,
    )
    if response_schema:
        return response.parsed
    return response.text


def _call_gemini_with_text(
    *,
    user_prompt: str,
    system_prompt_path: str,
    response_schema: Optional[Type[T]] = None,
    model: str = "gemini-2.5-pro",
):
    system_instruction = format_prompt(system_prompt_path)
    generation_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
    )
    if response_schema:
        generation_config.response_schema = response_schema
        generation_config.response_mime_type = "application/json"
    contents = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_prompt)]
    )
    response = google_client.models.generate_content(
        model=model,
        contents=contents,
        config=generation_config,
    )

    print(response)  # just print for now later process and return
