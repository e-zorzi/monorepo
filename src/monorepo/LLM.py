"""
Author: e-zorzi
License: Apache 2.0
"""

from google import genai
import os
import json
from datetime import datetime
from abc import ABC, abstractmethod
import base64
from io import BytesIO
import openai
from attrs import define, field
from typing import Optional, Union, Iterable
import numpy as np
from PIL import Image
from colorama import Fore, init as colorama_init
from uuid import uuid4
from monorepo.utils import load_api_keys

colorama_init(autoreset=True)

# Per request. The maximum daily spend (in terms of input tokens) will be
# ((N_tokens / 1_000_000) * RPD * price_per_1M) e.g. for Gemini2.5-pro,
#  which costs around 2$ per 1M tokens (Nov 2025), with this the max daily
# cost will be (20_000/1_000_000) * 10_000 * 2 = $400
_SAFEGUARD_N_TOKENS = 20_000

# Using the very-handwavy 4 letters = 1 token
_SAFEGUARD_N_LETTERS = _SAFEGUARD_N_TOKENS * 4

# For images
_SAFEGUARD_IMAGE_RESOLUTION = 1024

# GROQ Multimodal models

_VALID_GROQ_MULTIMODAL_MODELS = [
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-guard-4-12b",
]
_VALID_GROQ_DATE = "2025-12-16"


def _warn_requires_vllm(classname, model_id):
    print(
        Fore.YELLOW
        + f"[WARN] `{classname}` requires a connection with a local VLLM server. \
Make sure to run the command `vllm serve {model_id} <options>` in a terminal, and wait for its initialization."
        + Fore.WHITE
    )


def _warn_prompt_too_long(len_prompt, safeguard_length):
    print(
        Fore.YELLOW
        + f"[WARN] The passed prompt has length {len_prompt}, greater than the maximum allowed: {safeguard_length}. It will be truncated accordingly.\
If you want to increase this limit, change the constant _SAFEGUARD_N_LETTERS in the file from which you import this class."
        + Fore.WHITE
    )


def _warn_missing_key(key_name):
    print(Fore.RED + f"[ERROR] {key_name} is missing in the environment." + Fore.WHITE)


def encode_image_b64(image, format):
    im_file = BytesIO()
    image.save(im_file, format=format.upper())
    im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
    return base64.b64encode(im_bytes).decode("utf-8")


def get_batch_result(
    info_file_path: Union[str, os.PathLike],
) -> tuple[bool, Optional[str]]:
    load_api_keys()
    with open(info_file_path, "r") as read_handle:
        batch_name = read_handle.readlines()[0]
        batch_name.rstrip("\n ")
        assert batch_name.startswith("batches"), "Wrong file passed ?!"

    client = genai.Client()
    batch_job = client.batches.get(name=batch_name)  # Initial get

    # while batch_job.state.name not in completed_states:
    #     print(f"Current state: {batch_job.state.name}")
    #     time.sleep(30)  # Wait for 30 seconds before polling again
    #     batch_job = client.batches.get(name=job_name)

    # print(f"Job finished with state: {batch_job.state.name}")
    if batch_job.state.name == "JOB_STATE_FAILED":
        print(Fore.RED + f"[ERROR] {batch_job.error}" + Fore.WHITE)
        return (False, None)
    elif batch_job.state.name == "JOB_STATE_SUCCEEDED":
        result_file_name = batch_job.dest.file_name
        # Create new file name:
        # 1) Find last '.' by reverting the string
        last_dot_index = info_file_path[::-1].index(".")
        # 2) Compute the correct index
        last_dot_index = len(info_file_path) - last_dot_index - 1
        # 3) This is the path without the extension after "."
        path_no_extension = info_file_path[:last_dot_index]
        # 4) This is the final path
        results_file_path = f"{path_no_extension}.results.jsonl"
        print(
            Fore.GREEN
            + f"[SUCCESS] Downloading result file content to {results_file_path} ..."
            + Fore.WHITE
        )
        file_content = client.files.download(file=result_file_name)
        # Process file_content (bytes) as needed
        with open(results_file_path, "a") as write_handle:
            lines = file_content.decode("utf-8").split("\n")
            for line in lines:
                if line != "":
                    write_handle.write(json.dumps(json.loads(line)))
                    write_handle.write("\n")
        return (True, results_file_path)
    elif batch_job.state.name == "JOB_STATE_PENDING":
        print(Fore.YELLOW + "[INFO] Job still pending" + Fore.WHITE)
        print(batch_job)
        return (False, None)
    else:
        print(batch_job)
        return (False, None)


class IRemoteLLM(ABC):
    @abstractmethod
    def ask(
        self,
        *,
        prompt: str,
        images: Iterable[Union[np.ndarray, Image.Image]],
        **kwargs,
    ) -> str:
        pass


@define(kw_only=True, auto_attribs=True)
class GeminiLLM(IRemoteLLM):
    model_id: str
    api_key: str = field(default=None, repr=lambda _: "<|CENSORED|>")
    _delay: float = field(default=0.1)
    include_thoughts: bool = field(default=False)
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.95)
    aspect_ratio: str = field(default="1:1")
    image_size: str = field(default="1k")

    @aspect_ratio.validator
    def _aspect_ratio_check(self, attr, val):
        if val not in ["1:1", "16:9", "4:3", "3:4", "9:16", "2:3", "3:2"]:
            raise ValueError()

    @image_size.validator
    def _image_size_check(self, attr, val):
        if val not in ["1k", "2k", "4k"]:
            raise ValueError()

    def __attrs_post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv(
                "GEMINI_API_KEY",
            )

        self._client = genai.Client(api_key=self.api_key)

    def _get_config(
        self,
        thinking_budget=None,
        aspect_ratio=None,
        image_size=None,
        include_thoughts=None,
        generate_images: bool = False,
        as_dict: bool = False,
    ):
        if include_thoughts is None:
            _include_thoughts = include_thoughts
        else:
            _include_thoughts = self.include_thoughts
        if thinking_budget is None:
            thinking_config = (
                genai.types.ThinkingConfig(include_thoughts=_include_thoughts)
                if not as_dict
                else dict(include_thoughts=_include_thoughts)
            )
        else:
            thinking_config = (
                genai.types.ThinkingConfig(
                    include_thoughts=_include_thoughts,
                    thinking_budget=thinking_budget,
                )
                if not as_dict
                else dict(
                    include_thoughts=_include_thoughts,
                    thinking_budget=thinking_budget,
                )
            )

        _TYPE = genai.types.GenerateContentConfig if not as_dict else dict

        if generate_images:
            # Generate image config
            aspect_ratio = (
                aspect_ratio if aspect_ratio is not None else self.aspect_ratio
            )
            self._aspect_ratio_check("aspect_ratio", aspect_ratio)
            image_size = image_size if image_size is not None else self.image_size
            self._image_size_check("image_size", image_size)

            image_config = (
                genai.types.ImageConfig(
                    aspect_ratio=aspect_ratio, image_size=image_size
                )
                if not as_dict
                else dict(aspect_ratio=aspect_ratio, image_size=image_size)
            )
            if generate_images:
                response_modalities = ["TEXT", "IMAGE"]
            else:
                response_modalities = ["TEXT"]

            return _TYPE(
                temperature=self.temperature,
                top_p=self.top_p,
                thinking_config=thinking_config,
                image_config=image_config,
                response_modalities=response_modalities,
            )
        else:
            return _TYPE(
                temperature=self.temperature,
                top_p=self.top_p,
                thinking_config=thinking_config,
            )

    def _image_text_chat(
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
            _warn_prompt_too_long(len(prompt), _SAFEGUARD_N_LETTERS)

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

    def _text_chat(
        self,
        prompt,
        thinking_budget=None,
        return_metadata: bool = False,
    ):
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            _warn_prompt_too_long(len(prompt), _SAFEGUARD_N_LETTERS)

        response = self._client.models.generate_content(
            model=self.model_id,
            contents=[prompt[:_SAFEGUARD_N_LETTERS]],
            config=self._get_config(thinking_budget),
        )
        if return_metadata:
            return (response.text, response.usage_metadata)
        else:
            return response.text

    def ask(
        self,
        *,
        prompt: str,
        images: Iterable[Union[np.ndarray, Image.Image]] = None,
        thinking_budget=None,
        return_metadata: bool = False,
        **kwargs,
    ) -> str:
        """Primary method for generating (multimodal or unimodal) requests

        Args:
            prompt (str): text prompt
            images (Iterable[Union[np.ndarray, &quot;Image&quot;]], optional): a set of images related to the prompt, if a multimodal chat is required

        Returns:
            str: the response of the model
        """
        if images is not None:
            assert len(images) == 1, "Only 1 image is supported at the moment"
            return self._image_text_chat(
                prompt,
                images[0],
                thinking_budget=thinking_budget,
                return_metadata=return_metadata,
                **kwargs,
            )
        else:
            return self._text_chat(
                prompt,
                thinking_budget=thinking_budget,
                return_metadata=return_metadata,
                **kwargs,
            )

    def ask_for_later(
        self,
        *,
        prompt: str,
        images: Union[
            Iterable[Union[np.ndarray, Image.Image]], Iterable[Union[str, os.PathLike]]
        ] = None,
        generate_images: bool = False,
        thinking_budget: int = None,
        aspect_ratio: str = None,
        image_size: str = None,
        id: str = None,
        **kwargs,
    ):
        """Primary method for generating batch requests

        Args:
            prompt (str): text prompt
            images (Union[Iterable[Union[np.ndarray, PIL.Image]],Iterable[Union[str, Pathlike]]], optional):
                    a set of images related to the prompt (or paths to images), if a multimodal chat is required

        Returns:
            dict: a JSON-dumpable dictionary to be ingested by a later Gemini batch job (e.g. by saving it inside a JSONL file)
        """
        if id is None:
            id = uuid4().hex
        version = 1

        img_paths = []
        if images is not None:
            assert images[0], "images must be an iterable"
            # If images are passed as Images or np.arrays, then we save them locally for later upload
            if isinstance(images[0], np.ndarray) or isinstance(images[0], Image.Image):
                dir_path = os.path.join("/tmp", "batch_images")
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                for image in images:
                    if isinstance(image, np.ndarray):
                        image = Image.fromarray(image)
                        image_format = "png"
                    else:
                        image_format = image.format

                    img_path = os.path.join(
                        dir_path, f"{uuid4().hex}.{image_format.lower()}"
                    )
                    image.save(img_path, format=image_format.lower())
                    img_paths.append(img_path)
            elif isinstance(images[0], str) or isinstance(images[0], os.PathLike):
                img_paths.extend([str(p) for p in images])
            else:
                raise TypeError("Wrong image type")

        generation_config = self._get_config(
            thinking_budget, aspect_ratio, image_size, generate_images, as_dict=True
        )

        return dict(
            version=version,
            id=id,
            prompt=prompt,
            n_imgs=len(img_paths),
            img_paths=img_paths,
            generation_config=generation_config,
        )

    def submit_batch(
        self,
        jsonl_file_path: Union[str, os.PathLike],
    ) -> str:
        """Submits a batch request to Gemini via a JSONL file

        Args:
            jsonl_file_path (Union[str, os.PathLike]): path to a JSONL file containing a 'version 1'
                                                       object per line (obtained from `ask_for_later`)

        Raises:
            NotImplementedError: if any request-like object have 'version' > 1 (not supported at the moment)

        Returns:
            str: the path to the file containing info about the submitted batch (can be use for retrieval later)
        """
        now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        uuid = uuid4().hex

        cur_name = str(now) + "-" + uuid[:3] + uuid[-3:]

        if not os.path.exists(".batches"):
            os.mkdir(".batches")
        new_file_path = os.path.join(".batches", cur_name)

        jobs = []

        with open(jsonl_file_path, "r") as read_handle:
            for line in read_handle.readlines():
                jobs.append(json.loads(line))

        with open(f"{new_file_path}.jsonl", "a") as write_handle:
            for job in jobs:
                # Well-formatted JSON request object to be directly sent to Gemini without transformations
                if "key" in job and "request" in job:
                    json.dump(
                        job,
                        write_handle,
                    )
                    write_handle.write("\n")
                elif job["version"] == 1:
                    # Our custom request-like object that has to be transformed.
                    # Why we support a custom format? Because we can, by doing so,
                    # handle image uploading + batch processing in two phases, avoiding to
                    # include images in base64 format (very large) inside the requests directly
                    id = job["id"]
                    generation_config = job["generation_config"]
                    contents = [{"parts": [], "role": "user"}]
                    if job["n_imgs"] > 0:
                        for img_path in job["img_paths"]:
                            # Need to upload the image to Gemini and get the file path
                            uploaded_file = self._client.files.upload(file=img_path)

                            # Safer to do it like this
                            image_format = Image.open(img_path).format

                            contents[0]["parts"].append({
                                "fileData": {
                                    "fileUri": f"https://generativelanguage.googleapis.com/{uploaded_file.name}",
                                    "mimeType": f"image/{image_format.lower()}",
                                }
                            })
                        print(
                            Fore.CYAN
                            + f"[INFO] Uploaded {job['n_imgs']} images using Gemini Files API"
                        )
                    request_dict = dict(
                        contents=contents, generation_config=generation_config
                    )
                    # Append text prompt (always present)
                    contents[0]["parts"].append({"text": job["prompt"]})
                    new_json_line = dict(key=id, request=request_dict)

                    json.dump(
                        new_json_line,
                        write_handle,
                    )
                    write_handle.write("\n")
                else:
                    raise NotImplementedError

        batch_job = self._client.files.upload(
            file=f"{new_file_path}.jsonl",
            config=genai.types.UploadFileConfig(
                display_name=f"my-batch-requests-{cur_name}", mime_type="jsonl"
            ),
        )
        file_batch_job = self._client.batches.create(
            model=self.model_id,
            src=batch_job.name,
            config={"display_name": f"my-batch-requests-{cur_name}"},
        )

        print(f"[INFO] Created batch job: {file_batch_job.name}")
        with open(f"{new_file_path}.info", "w") as info_write_handle:
            info_write_handle.write(file_batch_job.name)
        return f"{new_file_path}.info"


@define(kw_only=True, auto_attribs=True)
class OpenAILLM(IRemoteLLM):
    model_id: str
    api_key: str = field(default=None, repr=lambda _: "<|CENSORED|>")
    _url: str = field(default="https://api.openai.com/v1")
    _delay: float = field(default=0.1)
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.95)
    max_output_length: int = field(default=12000)

    def __attrs_post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY")
        try:
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)
        except openai.OpenAIError as e:
            _warn_missing_key("OPENAI_API_KEY")
            raise e

    def _image_text_chat(self, prompt, image, **kwargs):
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
            _warn_prompt_too_long(len(prompt), _SAFEGUARD_N_LETTERS)

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
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format.lower()};base64,{image_bytes}"
                            },
                        },
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

    def _text_chat(
        self,
        prompt,
        **kwargs,
    ):
        if len(prompt) > _SAFEGUARD_N_LETTERS:
            _warn_prompt_too_long(len(prompt), _SAFEGUARD_N_LETTERS)

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

    def ask(
        self,
        *,
        prompt: str,
        images: Iterable[Union[np.ndarray, Image.Image]] = None,  # type: ignore
        **kwargs,
    ) -> str:
        """_summary_

        Args:
            prompt (str): text prompt
            images (Iterable[Union[np.ndarray, &quot;Image&quot;]], optional): a set of images related to the prompt, if a multimodal chat is required

        Returns:
            str: the response of the model
        """
        if images is not None:
            assert len(images) == 1, "Only 1 image is supported at the moment"
            return self._image_text_chat(
                prompt,
                images[0],
                **kwargs,
            )
        else:
            return self._text_chat(
                prompt,
                **kwargs,
            )


@define(kw_only=True, auto_attribs=True)
class CerebrasLLM(OpenAILLM):
    api_key: str = field(default=None, repr=lambda _: "<|CENSORED|>")
    _url: str = field(default="https://api.cerebras.ai/v1")

    def __attrs_post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("CEREBRAS_API_KEY")

        try:
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)
        except openai.OpenAIError:
            _warn_missing_key("CEREBRAS_API_KEY")
            raise openai.OpenAIError(
                "The api_key client option must be set either by passing api_key to the client or by setting the CEREBRAS_API_KEY environment variable"
            )

    def _image_text_chat(
        self,
        prompt,
        image,
    ):
        raise NotImplementedError(
            "Cerebras doesn't support multimodal serving at the moment. Try another class like `OpenAILLM' or `GeminiLLM'"
        )


@define(kw_only=True, auto_attribs=True)
class GroqLLM(OpenAILLM):
    api_key: str = field(default=None, repr=lambda _: "<|CENSORED|>")
    _url: str = field(default="https://api.groq.com/openai/v1")

    def __attrs_post_init__(self):
        if self.api_key is None:
            self.api_key = os.getenv("GROQ_API_KEY")
        try:
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)
        except openai.OpenAIError:
            _warn_missing_key("GROQ_API_KEY")
            raise openai.OpenAIError(
                "The api_key client option must be set either by passing api_key to the client or by setting the GROQ_API_KEY environment variable"
            )

    def _image_text_chat(
        self,
        prompt,
        image,
    ):
        if self.model_id not in _VALID_GROQ_MULTIMODAL_MODELS:
            raise ValueError(
                f"Groq only supports the following models for multimodal serving: 'meta-llama/llama-4-maverick-17b-128e-instruct', 'meta-llama/llama-4-scout-17b-16e-instruct' and 'meta-llama/llama-guard-4-12b' (as of {_VALID_GROQ_DATE})"
            )
        return super()._image_text_chat(prompt, image)


@define(kw_only=True, auto_attribs=True)
class ClientBasedLLM(OpenAILLM):
    api_key: str = field(default="EMPTY")
    _port: int = field(default=8000)
    _url: Optional[str] = None

    def __attrs_post_init__(self):
        _warn_requires_vllm(self.__class__.__name__, self.model_id)

        if self._url is None:
            self._url = f"http://localhost:{self._port}/v1"

        self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)


# Need to think about if it is worth implementing this, to avoid requiring a VLLM dependency
@define(kw_only=True, auto_attribs=True)
class LocalLLM(IRemoteLLM):
    model_id: str
    temperature: float = field(default=1.0)
    top_p: float = field(default=0.95)

    def __attrs_post_init__(self):
        # self._sampling_params =
        try:
            from vllm import LLM
        except (ImportError, ModuleNotFoundError) as e:
            print(
                Fore.RED + "[ERROR] `LocalLLM` requires the library `vllm`. Install it."
            )
            raise e

    def _text_chat(self, prompt):
        raise NotImplementedError

    def _image_text_chat(self, prompt, image, **kwargs):
        raise NotImplementedError

    def ask(self, *, prompt, images, **kwargs):
        return super().ask(prompt=prompt, images=images, **kwargs)
