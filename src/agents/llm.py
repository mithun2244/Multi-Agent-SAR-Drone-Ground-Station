"""A thin NVIDIA NIM client — text and vision, stdlib only.

All LLM- and VLM-backed agents (Scene, the Path briefing, and later Health and
Interview) run against NVIDIA NIM at https://integrate.api.nvidia.com/v1.

NIM speaks the OpenAI chat-completions shape, so this is one HTTPS POST with
`urllib` — no `openai` package, no SDK, dependency count unchanged at three.

Vision models on NIM take the image inline in the message content as an
`<img src="data:...;base64,...">` tag rather than OpenAI's content-array form.
That is NVIDIA's documented shape for `meta/llama-3.2-*-vision-instruct`, and it
is what `complete(prompt, image=...)` builds.

Every agent takes its completer as a callable, so tests and demos inject a stub
and the suite never depends on a network, a key, or someone's quota. `from_env`
returns None rather than raising when `NVIDIA_API_KEY` is unset — an agent that
cannot reach a model must say so, not fail the search.
"""

import base64
import json
import os
import urllib.error
import urllib.request

API_ROOT = "https://integrate.api.nvidia.com/v1"
DEFAULT_LLM_MODEL = "meta/llama-3.1-70b-instruct"
DEFAULT_VLM_MODEL = "meta/llama-3.2-90b-vision-instruct"
# For narrow, high-volume extraction where latency matters more than reasoning.
FAST_LLM_MODEL = "meta/llama-3.1-8b-instruct"

# NIM rejects inline images past roughly this size and wants its assets API
# instead. ponytail: refuse loudly rather than half-implement the upload flow —
# the drone's frames are well under it, and a silent truncation would mean the
# model describing a picture nobody sent.
MAX_INLINE_IMAGE_BYTES = 180_000


class LLMUnavailable(RuntimeError):
    """The model could not be reached or refused to answer."""


class NimClient:
    """Minimal NVIDIA NIM chat-completions client."""

    def __init__(
        self,
        api_key,
        model=DEFAULT_LLM_MODEL,
        vision_model=DEFAULT_VLM_MODEL,
        timeout=30.0,
        api_root=API_ROOT,
        temperature=0.2,
        max_tokens=512,
    ):
        if not api_key:
            raise ValueError("an API key is required")
        self.api_key = api_key
        self.model = model
        self.vision_model = vision_model
        self.timeout = timeout
        self.api_root = api_root.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.calls = 0

    @classmethod
    def from_env(cls, **kwargs):
        """Build from NVIDIA_API_KEY, or None when it is not set."""
        key = os.environ.get("NVIDIA_API_KEY")
        return cls(key, **kwargs) if key else None

    def using(self, model=None, vision_model=None, **kwargs):
        """A sibling client on different models, sharing the key and settings.

        Agents pick their own model — Interview wants the small fast one, Path
        wants the large one — without each rebuilding a client from the
        environment.
        """
        settings = {
            "vision_model": vision_model or self.vision_model,
            "timeout": self.timeout,
            "api_root": self.api_root,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        settings.update(kwargs)
        return NimClient(self.api_key, model=model or self.model, **settings)

    def complete(self, prompt, image=None, mime_type="image/jpeg"):
        """One turn in, text out. Passing `image` bytes switches to the VLM.

        `image` may also be a sequence of frames — the Scene agent sends a crop
        of the subject and the whole frame. They inline in the order given, and
        the size limit is on the *total*: NIM's cap is on the request, not on
        each picture in it.
        """
        content = prompt
        model = self.model
        images = [image] if isinstance(image, (bytes, bytearray)) else list(image or ())
        if images:
            total = sum(len(each) for each in images)
            if total > MAX_INLINE_IMAGE_BYTES:
                raise LLMUnavailable(
                    f"{len(images)} image(s) total {total} bytes, over NIM's "
                    f"{MAX_INLINE_IMAGE_BYTES} inline limit; upload via the assets "
                    f"API instead"
                )
            tags = "".join(
                f' <img src="data:{mime_type};base64,{base64.b64encode(each).decode()}" />'
                for each in images
            )
            content = f"{prompt}{tags}"
            model = self.vision_model

        request = urllib.request.Request(
            f"{self.api_root}/chat/completions",
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            raise LLMUnavailable(f"{model}: {e}") from e

        self.calls += 1
        return _first_text(payload, model)


def _first_text(payload, model):
    """The assistant's text out of a chat-completions response."""
    choices = payload.get("choices") or []
    if not choices:
        # An empty or filtered response is not text. Returning "" here would let
        # a caller publish a blank briefing as though the model had written one.
        raise LLMUnavailable(f"{model}: no choices ({payload.get('error', payload)})")
    text = (choices[0].get("message") or {}).get("content") or ""
    text = text.strip()
    if not text:
        raise LLMUnavailable(f"{model}: empty completion")
    return text


def static_completer(reply):
    """A completer that always returns the same text. For tests and demos."""
    def complete(prompt, image=None, mime_type="image/jpeg"):
        return reply
    return complete


def unavailable_completer(reason="no API key configured"):
    """A completer that always fails, to exercise the degraded path."""
    def complete(prompt, image=None, mime_type="image/jpeg"):
        raise LLMUnavailable(reason)
    return complete
