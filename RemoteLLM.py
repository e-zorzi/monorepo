from google import genai
import os
from abc import ABC, abstractmethod
import base64
from io import BytesIO
import openai
from attrs import define, field

PROMPT_WITH_CLASS = (
    "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object/scene based on the user's commands.\
You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image, which contains a {OBJCLASS}. \
Reason about the {OBJCLASS} that you are seeing in the image, comparing what you know about the task (given the user commands) and the given scene. For example, \
if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. At the end of the reasoning process, give it a score from 1 to 5 on how much you are sure about the fact that the current observation \
(given by the image) matches the target task given by the user, where 1 means 'extremely unlikely' and 5 'extremely likely'. Be concise (use at most 100 words) and return a response with \
 the following form: <motivation>Your motivations</motivation><score>Your score</score>"
)

PROMPT = (
    "You are a robot navigating an enclosed space. Your goal is to navigate to the correct object/scene based on the user's commands.\
You were given the following task by the user '{USER_TASK}'. Currently, you are facing a scene represented by the given image.\
Reason about what you are seeing, comparing what you know about the task (given the user commands) and the given scene. For example, \
if the task is 'Navigate to the black leather sofa near a lampstand' your reasoning process will be 'I'm currently observing a brown sofa \
which is different than black, making it unlikely to be the target sofa. Moreover, there is no lampstand near it, only a rug and a window' \
etc. If there are distortions or artifact, do not focus on them, focus on the object at hand. At the end of the reasoning process, give it a score from 1 to 5 on how much you are sure about the fact that the current observation \
(given by the image) matches the target task given by the user, where 1 means 'extremely unlikely' and 5 'extremely likely'. Be concise (use at most 100 words) and return a response with \
 the following form: <motivation>Your motivations</motivation><score>Your score</score>"
)

PROMPT_ASK_FOR_PROPERTIES = "Currently, you are facing a scene represented by the given image, which contains a {OBJCLASS}. If there are distortions or artifact, do not focus on them, focus on the object at hand. \
Answer these three questions: \
    1. What color is the {OBJCLASS} that you are seeing? 2. What shape/dimensions does it have? 3. What other objects are near? \
    Return an asnwer formatted like this <color>color answer</color>\n<shape>shape and dimension answer</shape>\n<objects>objects answer</objects>\n"


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
    api_key: str = field(default=os.getenv("GEMINI_API_KEY", "NONE"))
    _client: genai.Client = None
    _delay: float = field(default=0.1)
    _temperature: float = field(default=1.0)
    _top_p: float = field(default=0.95)

    def __attrs_post_init__(self):
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)

    def _get_config(self, thinking_budget=None):
        if thinking_budget is None:
            thinking_config = genai.types.ThinkingConfig(include_thoughts=True)
        else:
            thinking_config = genai.types.ThinkingConfig(
                include_thoughts=True, thinking_budget=thinking_budget
            )
        return genai.types.GenerateContentConfig(
            temperature=self._temperature,
            top_p=self._top_p,
            thinking_config=thinking_config,
        )

    def image_text_chat(self, prompt, image, thinking_budget=None):
        image_bytes = BytesIO()

        # Save the PIL image to the byte stream in JPEG format.
        # You can use other formats like 'PNG' as well.
        # For JPEG, you can also specify the quality, e.g., img.save(image_bytes_io, format='JPEG', quality=90)
        image.save(image_bytes, format="PNG")
        image_bytes = image_bytes.getvalue()

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[
                genai.types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/png",
                ),
                prompt,
            ],
            config=self._get_config(thinking_budget),
        )
        return response.text

    def text_chat(self, prompt, thinking_budget=None):
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=([prompt],),
            config=self._get_config(thinking_budget),
        )
        return response.text


@define(kw_only=True, auto_attribs=True)
class OpenAILLM(IRemoteLLM):
    model_id: str
    api_key: str = field(default=os.getenv("OPENAI_API_KEY", "NONE"))
    _url: str = field(default="https://api.openai.com/v1")
    _client: openai.OpenAI = None
    _delay: float = field(default=0.1)
    _temperature: float = field(default=1.0)
    _top_p: float = field(default=0.95)

    def __attrs_post_init__(self):
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api_key, base_url=self._url)

    def _encode_image_from_path(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _encode_image(self, image):
        im_file = BytesIO()
        image.save(im_file, format="JPEG")
        im_bytes = im_file.getvalue()  # im_bytes: image in binary format.
        return base64.b64encode(im_bytes).decode("utf-8")

    def image_text_chat(self, prompt, image):
        image_bytes = self._encode_image(image)

        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_bytes}",
                        },
                    ],
                }
            ],
            stream=True,
        )
        stringbuilder = ""

        for chunk in completion:
            token = chunk.choices[0].delta.content
            if token:
                stringbuilder += f"{token} "

        return stringbuilder

    def text_chat(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
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
    api_key: str = field(default=os.getenv("CEREBRAS_API_KEY", "NONE"))
    _url: str = field(default="https://api.cerebras.ai/v1")

    def image_text_chat(self, prompt, image):
        raise NotImplementedError(
            "Cerebras still doesn't support multimodal serving. Try another class like `OpenAILLM' or `GeminiLLM'"
        )
