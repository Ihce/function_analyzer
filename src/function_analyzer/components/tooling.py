#!/usr/bin/env python3
# components/tooling.py
from typing import Any, Dict, Type

try:
    from pydantic import BaseModel  # v2 preferred
except ImportError as e:
    raise SystemExit("Missing dependency: pip install pydantic>=2") from e


def json_schema_for(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Pydantic v2 (preferred) with v1 fallback."""
    try:
        return model_cls.model_json_schema()  # pydantic v2
    except AttributeError:
        return model_cls.schema()             # pydantic v1


def build_tool_spec(*, name: str, description: str, ArgsModel: Type[BaseModel]) -> Dict[str, Any]:
    params = json_schema_for(ArgsModel)
    # tighten the schema a bit:
    if "additionalProperties" not in params:
        params["additionalProperties"] = False
    return {
        "name": name,
        "description": description,
        "parameters": params,
    }
