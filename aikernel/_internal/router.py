from typing import Literal, NotRequired, TypedDict

from litellm import Router

LLMModelAlias = Literal[
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "claude-3.5-sonnet",
    "claude-3.7-sonnet",
]
LLMModelName = Literal[
    "vertex_ai/gemini-2.0-flash",
    "vertex_ai/gemini-2.0-flash-lite",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
]

MODEL_ALIAS_MAPPING: dict[LLMModelAlias, LLMModelName] = {
    "gemini-2.0-flash": "vertex_ai/gemini-2.0-flash",
    "gemini-2.0-flash-lite": "vertex_ai/gemini-2.0-flash-lite",
    "claude-3.5-sonnet": "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "claude-3.7-sonnet": "bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0",
}


class RouterModelLitellmParams(TypedDict):
    model: str
    api_base: NotRequired[str]
    api_key: NotRequired[str]
    rpm: NotRequired[int]


class RouterModel(TypedDict):
    model_name: str
    litellm_params: RouterModelLitellmParams


def get_router(*, model_priority_list: list[LLMModelAlias]) -> Router:
    model_list: list[RouterModel] = [
        {"model_name": model, "litellm_params": {"model": MODEL_ALIAS_MAPPING[model]}} for model in model_priority_list
    ]
    fallbacks = [
        {model: [other_model for other_model in model_priority_list if other_model != model]}
        for model in model_priority_list
    ]
    return Router(model_list=model_list, fallbacks=fallbacks)
