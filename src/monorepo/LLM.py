from google import genai
import os
from abc import ABC, abstractmethod
import base64
from io import BytesIO
import openai
from attrs import define, field
from typing import Optional
import numpy as np
from PIL import Image

# Per request. The maximum daily spend (in terms of input tokens) will be
# ((N_tokens / 1_000_000) * RPD * price_per_1M) e.g. for Gemini2.5-pro,
#  which costs around 2$ per 1M tokens (Nov 2025), with this the max daily
# cost will be (20_000/1_000_000) * 10_000 * 2 = $400
_SAFEGUARD_N_TOKENS = 20_000

# Using the very-handwavy 4 letters = 1 token
_SAFEGUARD_N_LETTERS = _SAFEGUARD_N_TOKENS * 4

# For images
_SAFEGUARD_IMAGE_RESOLUTION = 1024


def encode_image_b64(image, format):
    im_file = BytesIO()
    image.save(im_file, format=format.upper())
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    return base64.b64encode(im_bytes).decode("utf-8")


class IRemoteLLM(ABC):
    @abstractmethod
    def image_text_chat(self, prompt: str, image, **kwargs):
        pass

    @abstractmethod
    def text_chat(self, prompt: str, **kwargs):
        pass


@define(kw_only=True, auto_attribs=True)
class GeminiLLM(IRemoteLLM):
    model_id: str
    api_key: str = None
    _delay: float = field(default=0.1)
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.95)

    def __attrs_post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("GEMINI_API_KEY")

        self._client = genai.Client(api_key=self.api_key)

    def _get_config(self, thinking_budget=None):
        if thinking_budget is None:
            thinking_config = genai.types.ThinkingConfig(include_thoughts=True)
        else:
            thinking_config = genai.types.ThinkingConfig(
                include_thoughts=True, thinking_budget=thinking_budget
            )
        return genai.types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            thinking_config=thinking_config,
        )

    def image_text_chat(
        self,
        prompt,
        image,
        thinking_budget=None,
        return_metadata: bool = False,
    ):
        # Handle arrays
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image_format = "png"
        else:
            image_format = image.format

        if image_format is None or image_format == "None":
            raise ValueError(
                "Wrong image format. I got 'None'. Check how you constructed the image."
            )

        # Safety checks
        height, width = image.size
        if height > _SAFEGUARD_IMAGE_RESOLUTION or width > _SAFEGUARD_IMAGE_RESOLUTION:
            raise ValueError(
                f"Image size safeguard: passed an image of resolution {width} x {height}, larger than the safeguard {_SAFEGUARD_IMAGE_RESOLUTION} x {_SAFEGUARD_IMAGE_RESOLUTION}"
            )
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            print(
                f"!! Warning !! The passed prompt has length {len(prompt)}, greater than the maximum allowed: {_SAFEGUARD_N_LETTERS}. It will be truncated accordingly."
            )

        image_format = image.format
        if image_format is None or image_format == "None":
            raise ValueError(
                "Wrong image format. I got 'None'. Check how you constructed the image."
            )

        image_bytes = BytesIO()
        image.save(image_bytes, format=image_format.upper())
        image_bytes = image_bytes.getvalue()

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[
                genai.types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=f"image/{image_format.lower()}",
                ),
                prompt[:_SAFEGUARD_N_LETTERS],
            ],
            config=self._get_config(thinking_budget),
        )

        if return_metadata:
            return (response.text, response.usage_metadata)
        else:
            return response.text

    def text_chat(
        self,
        prompt,
        thinking_budget=None,
        return_metadata: bool = False,
    ):
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            print(
                f"!! Warning !! The passed prompt has length {len(prompt)}, greater than the maximum allowed: {_SAFEGUARD_N_LETTERS}. It will be truncated accordingly."
            )
        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[prompt[:_SAFEGUARD_N_LETTERS]],
            config=self._get_config(thinking_budget),
        )
        if return_metadata:
            return (response.text, response.usage_metadata)
        else:
            return response.text


@define(kw_only=True, auto_attribs=True)
class OpenAILLM(IRemoteLLM):
    model_id: str
    api_key: str = None
    _url: str = field(default="https://api.openai.com/v1")
    _delay: float = field(default=0.1)
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.95)
    max_output_length: int = field(default=12000)

    def __attrs_post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)

    def image_text_chat(
        self,
        prompt,
        image,
    ):
        # Handle arrays
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            image_format = "png"
        else:
            image_format = image.format

        if image_format is None or image_format == "None":
            raise ValueError(
                "Wrong image format. I got 'None'. Check how you constructed the image."
            )

        # Safety checks
        height, width = image.size
        if height > _SAFEGUARD_IMAGE_RESOLUTION or width > _SAFEGUARD_IMAGE_RESOLUTION:
            raise ValueError(
                f"Image size safeguard: passed an image of resolution {width} x {height}, larger than the safeguard {_SAFEGUARD_IMAGE_RESOLUTION} x {_SAFEGUARD_IMAGE_RESOLUTION}"
            )
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            print(
                f"!! Warning !! The passed prompt has length {len(prompt)}, greater than the maximum allowed: {_SAFEGUARD_N_LETTERS}. It will be truncated accordingly."
            )

        image_format = image.format
        if image_format is None or image_format == "None":
            raise ValueError(
                "Wrong image format. I got 'None'. Check how you constructed the image."
            )

        image_bytes = encode_image_b64(image, image_format)
        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt[:_SAFEGUARD_N_LETTERS],
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format.lower()};base64,{image_bytes}"
                            },
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=int(self.max_output_length / 4),
            stream=True,
        )
        stringbuilder = ""

        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token:
                stringbuilder += f"{token}"

        return stringbuilder

    def text_chat(
        self,
        prompt,
    ):
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            print(
                f"!! Warning !! The passed prompt has length {len(prompt)}, greater than the maximum allowed: {_SAFEGUARD_N_LETTERS}. It will be truncated accordingly."
            )

        completion = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt[:_SAFEGUARD_N_LETTERS],
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=int(self.max_output_length / 4),
            stream=True,
        )
        stringbuilder = ""

        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token:
                stringbuilder += f"{token}"

        return stringbuilder


@define(kw_only=True, auto_attribs=True)
class CerebrasLLM(OpenAILLM):
    api_key: str = None
    _url: str = field(default="https://api.cerebras.ai/v1")

    def __attrs_post_init__(self):
        if self.api_key is None:
            if "CEREBRAS_API_KEY" not in os.environ:
                print("[WARNING] Missing CEREBRAS_API_KEY in environment!")
            self.api_key = os.getenv("CEREBRAS_API_KEY", "<|MISSING_API_KEY|>")

        self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)

    def image_text_chat(
        self,
        prompt,
        image,
    ):
        raise NotImplementedError(
            "Cerebras still doesn't support multimodal serving. Try another class like `OpenAILLM' or `GeminiLLM'"
        )


_VALID_GROQ_MULTIMODAL_MODELS = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-guard-4-12b",
]
_VALID_GROQ_DATE = "2025-12-16"


@define(kw_only=True, auto_attribs=True)
class GroqLLM(OpenAILLM):
    api_key: str = None
    _url: str = field(default="https://api.groq.com/openai/v1")

    def __attrs_post_init__(self):
        if self.api_key is None:
            if "GROQ_API_KEY" not in os.environ:
                print("[WARNING] Missing GROQ_API_KEY in environment!")
            self.api_key = os.getenv("GROQ_API_KEY", "<|MISSING_API_KEY|>")
        self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)

    def image_text_chat(
        self,
        prompt,
        image,
    ):
        if self.model_id not in _VALID_GROQ_MULTIMODAL_MODELS:
            raise ValueError(
                f"Groq only supports the following models for multimodal serving: 'meta-llama/llama-4-maverick-17b-128e-instruct', 'meta-llama/llama-4-scout-17b-16e-instruct' and 'meta-llama/llama-guard-4-12b' (as of {_VALID_GROQ_DATE})"
            )
        return super().image_text_chat(prompt, image)


@define(kw_only=True, auto_attribs=True)
class VllmLLM(OpenAILLM):
    api_key: str = "EMPTY"
    _port: int = field(default=8000)
    _url: Optional[str] = None

    def __attrs_post_init__(self):
        if self._url is None:
            self._url = f"http://localhost:{self._port}/v1"

        self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)


## Need to think about if it is worth implementing this, to avoid requiring a VLLM dependency
# @define(kw_only=True, auto_attribs=True)
# class VllmOfflineLLM(IRemoteLLM):
#     model_id: str
#     temperature: float = field(default=1.0)
#     top_p: float = field(default=0.95)

#     def __attrs_post_init__(self):
#         #self._sampling_params =

#     def text_chat(self, prompt):
#         return super().text_chat(prompt, )

#     def image_text_chat(self, prompt, image, **kwargs):
#         return super().image_text_chat(prompt, image, **kwargs)
