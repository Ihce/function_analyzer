#!/usr/bin/env python3
# agent.py — Orchestrates GPT-OSS (via Ollama) + your components tools using function-calling.

import json
import logging
from typing import Any, Dict, Optional, Tuple, Type

try:
    from openai import OpenAI  # OpenAI Python SDK >= 1.x
except ImportError as e:
    raise SystemExit("Missing dependency: pip install openai>=1.0.0") from e

from components import binary_loader as bl
from components import boundary_detector as bd
from components import disassembler as dz


class BinaryAnalysisAgent:
    """
    Keeps state in-memory; the LLM decides which tool to call. We validate args via Pydantic
    (from each component's ArgsModel) before invoking the real tool .run() methods.
    """

    def __init__(
        self,
        *,
        ollama_url: str = "http://localhost:11434/v1",
        model_id: str = "gpt-oss:20b",
        boundary_ckpt: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.client = OpenAI(base_url=ollama_url, api_key="ollama")
        self.model_id = model_id
        self.boundary_ckpt = boundary_ckpt
        self.log = logger or logging.getLogger("agent")
        self.log.setLevel(logging.INFO)

        # Instantiate tools (existing behavior unchanged)
        self.loader = bl.BinaryLoaderTool()
        self.boundaries = bd.BoundaryDetectorTool()
        self.disasm = dz.DisassemblerTool()

        # Self-described function schemas from components
        self.functions = [
            bl.BinaryLoaderTool.tool_spec(),
            bd.BoundaryDetectorTool.tool_spec(),
            dz.DisassemblerTool.tool_spec(),
        ]

        # State shared between tool calls
        self.state: Dict[str, Any] = {
            "binary": None,        # result from loader
            "functions": None,     # list from boundary detector
            "disassembly": None,   # list from disassembler
        }

        # Map tool -> (callable, ArgsModel)
        self._tools: Dict[str, Tuple[Any, Type]] = {
            bl.BinaryLoaderTool.name: (self._call_binary_loader, bl.BinaryLoaderTool.ArgsModel),
            bd.BoundaryDetectorTool.name: (self._call_boundary_detector, bd.BoundaryDetectorTool.ArgsModel),
            dz.DisassemblerTool.name: (self._call_disassembler, dz.DisassemblerTool.ArgsModel),
        }

        self._boundary_loaded = False

    # ---------- Validation helper ----------

    @staticmethod
    def _validate_args(args_raw: Dict[str, Any], ArgsModel: Type) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            obj = ArgsModel(**args_raw)  # pydantic v2
            try:
                return obj.model_dump(), None
            except AttributeError:
                return obj.dict(), None       # pydantic v1
        except Exception as e:
            return None, str(e)

    # ---------- Tool wrappers (thin adapters; your tool behavior untouched) ----------

    def _call_binary_loader(self, *, file_path: str) -> str:
        self.log.info(f"[binary_loader] file_path={file_path}")
        result = self.loader.run(file_path)
        if not result.get("success"):
            return json.dumps({"error": result.get("error", "Failed to load binary")})

        self.state["binary"] = result
        arch = result.get("architecture")
        base = int(result.get("base_address", 0))
        size = int(result.get("text_size", 0))
        return json.dumps({
            "summary": "Loaded .text successfully.",
            "arch": arch,
            "base_address": f"0x{base:08X}",
            "text_size": size,
        })

    def _ensure_boundary_model(self, model_path: Optional[str]) -> Optional[str]:
        if self._boundary_loaded:
            return None
        ckpt = model_path or self.boundary_ckpt
        if not ckpt:
            return "Boundary model checkpoint was not provided."
        try:
            self.log.info(f"[boundary_detector] loading model: {ckpt}")
            self.boundaries.load_model(ckpt)
            self._boundary_loaded = True
            return None
        except Exception as e:
            return f"Failed to load boundary model: {e}"

    def _call_boundary_detector(self, *, model_path: Optional[str] = None) -> str:
        if not self.state.get("binary") or not self.state["binary"].get("success"):
            return json.dumps({"error": "No binary loaded yet."})

        maybe_err = self._ensure_boundary_model(model_path)
        if maybe_err:
            return json.dumps({"error": maybe_err})

        text = self.state["binary"]["text_array"]
        base = int(self.state["binary"]["base_address"])
        self.log.info(f"[boundary_detector] running on {len(text)} bytes, base=0x{base:08X}")

        result = self.boundaries.run(text, base)
        if not result.get("success"):
            return json.dumps({"error": result.get("error", "Boundary detection failed")})

        funcs = result.get("functions", [])
        self.state["functions"] = funcs
        count = int(result.get("total_functions", len(funcs)))
        examples = [hex(f.get("start_address", 0)) for f in funcs[:3]]
        return json.dumps({
            "summary": f"Detected {count} functions.",
            "count": count,
            "examples": examples,
        })

    def _call_disassembler(self) -> str:
        if not self.state.get("binary") or not self.state.get("functions"):
            return json.dumps({"error": "No functions to disassemble."})

        text = self.state["binary"]["text_array"]
        base = int(self.state["binary"]["base_address"])
        arch = self.state["binary"]["architecture"]
        funcs = self.state["functions"]

        self.log.info(f"[disassembler] {len(funcs)} functions, base=0x{base:08X}, arch={arch}")
        result = self.disasm.run(text, funcs, base, arch)
        if not result.get("success"):
            return json.dumps({"error": result.get("error", "Disassembly failed")})

        disassembled = result.get("disassembled_functions", [])
        self.state["disassembly"] = disassembled
        n = int(result.get("total_functions", len(disassembled)))

        example_preview = None
        if disassembled:
            f0 = disassembled[0]
            addr = hex(f0["function"]["start_address"])
            preview = (f0.get("disassembly_preview") or "").strip()
            example_preview = {"addr": addr, "preview": preview}

        return json.dumps({
            "summary": f"Disassembled {n} functions.",
            "total": n,
            "example": example_preview,
        })

    # ---------- Agent loop ----------

    def analyze(self, file_path: str, *, system_prompt: Optional[str] = None, max_iters: int = 8) -> str:
        messages = [
            {"role": "system", "content": system_prompt or "You are a concise binary analyst."},
            {"role": "user", "content": f"Analyze this binary and explain what it does: {file_path}"},
        ]
        path_hint = file_path

        for i in range(max_iters):
            self.log.info(f"[agent] iteration {i+1}")
            resp = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                functions=self.functions,
                function_call="auto",
            )
            msg = resp.choices[0].message

            # Tool call?
            if getattr(msg, "function_call", None):
                name = msg.function_call.name
                raw = msg.function_call.arguments or "{}"
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = {}

                # Friendly autofill on first loader call
                if name == bl.BinaryLoaderTool.name and "file_path" not in parsed:
                    parsed["file_path"] = path_hint

                func, ArgsModel = self._tools.get(name, (None, None))
                if func is None:
                    content = json.dumps({"error": f"Unknown tool: {name}"})
                else:
                    kwargs, err = self._validate_args(parsed, ArgsModel)
                    if err:
                        content = json.dumps({"error": f"Invalid arguments: {err}"})
                    else:
                        content = func(**kwargs)  # type: ignore[arg-type]

                messages.append({"role": "function", "name": name, "content": content})
                continue

            # Final answer (no more tools)
            final = msg.content or ""
            self.log.info("[agent] final answer received.")
            return final.strip()

        return "Stopped after max iterations without a final answer."
