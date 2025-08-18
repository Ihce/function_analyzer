#!/usr/bin/env python3
# agent.py — Orchestrates GPT-OSS (via Ollama) + your components tools using function-calling.

import json
import logging
from typing import Any, Dict, Optional, Tuple, Type

try:
    from openai import OpenAI  # OpenAI Python SDK >= 1.x
except ImportError as e:
    raise SystemExit("Missing dependency: pip install openai>=1.0.0") from e

from .components import binary_loader as bl
from .components import boundary_detector as bd
from .components import disassembler as dz


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
        verbose: bool = True,  # Add verbose flag for detailed logging
        log_file: Optional[str] = "agent_analysis.log",  # Log file path
        log_to_console: bool = True,  # Also log to console
    ) -> None:
        self.client = OpenAI(base_url=ollama_url, api_key="ollama")
        self.model_id = model_id
        self.boundary_ckpt = boundary_ckpt
        self.verbose = verbose

        # Set up logging
        self.log = logger or logging.getLogger("agent")
        if self.verbose:
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.INFO)

        # Clear existing handlers to avoid duplicates
        self.log.handlers = []

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - [%(levelname)s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter('[%(levelname)s] %(message)s')

        # Add file handler if log_file is specified
        if log_file:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(detailed_formatter)
            self.log.addHandler(file_handler)

            # Log session start
            self.log.info("="*80)
            self.log.info(f"NEW ANALYSIS SESSION - Model: {model_id}")
            self.log.info(f"Ollama URL: {ollama_url}")
            self.log.info("="*80)

        # Add console handler if requested
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(simple_formatter)
            self.log.addHandler(console_handler)

        # Instantiate tools
        self.loader = bl.BinaryLoaderTool()
        self.boundaries = bd.BoundaryDetectorTool()
        self.disasm = dz.DisassemblerTool()

        # Self-described function schemas from components
        self.functions = [
            bl.BinaryLoaderTool.tool_spec(),
            bd.BoundaryDetectorTool.tool_spec(),
            dz.DisassemblerTool.tool_spec(),
        ]

        self.log.debug(f"Available tools: {[f['name'] for f in self.functions]}")

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

    # ---------- Tool wrappers ----------

    def _call_binary_loader(self, *, file_path: str) -> str:
        self.log.info(f"[TOOL EXEC] binary_loader with file_path={file_path}")
        result = self.loader.run(file_path)

        if not result.get("success"):
            error_msg = result.get("error", "Failed to load binary")
            self.log.error(f"[TOOL RESULT] binary_loader failed: {error_msg}")
            return json.dumps({"error": error_msg})

        self.state["binary"] = result
        arch = result.get("architecture")
        base = int(result.get("base_address", 0))
        size = int(result.get("text_size", 0))

        # Log detailed binary info
        self.log.info("="*60)
        self.log.info("[BINARY LOADER RESULTS]")
        self.log.info(f"  File: {result.get('file_path')}")
        self.log.info(f"  Architecture: {arch}")
        self.log.info(f"  Base Address: 0x{base:08X}")
        self.log.info(f"  Image Base: 0x{result.get('image_base', 0):08X}")
        self.log.info(f"  Section RVA: 0x{result.get('section_rva', 0):08X}")
        self.log.info(f"  Text Size: {size} bytes")
        self.log.info(f"  Is 64-bit: {result.get('is_64bit')}")
        self.log.info("="*60)

        success_msg = {
            "summary": "Loaded .text successfully.",
            "arch": arch,
            "base_address": f"0x{base:08X}",
            "text_size": size,
        }
        return json.dumps(success_msg)

    def _ensure_boundary_model(self, model_path: Optional[str]) -> Optional[str]:
        if self._boundary_loaded:
            return None
        ckpt = model_path or self.boundary_ckpt
        if not ckpt:
            return "Boundary model checkpoint was not provided."
        try:
            self.log.info(f"[MODEL] Loading boundary detector model: {ckpt}")
            self.boundaries.load_model(ckpt)
            self._boundary_loaded = True
            return None
        except Exception as e:
            return f"Failed to load boundary model: {e}"

    def _call_boundary_detector(self, *, model_path: Optional[str] = None) -> str:
        self.log.info(f"[TOOL EXEC] boundary_detector")

        if not self.state.get("binary") or not self.state["binary"].get("success"):
            error_msg = "No binary loaded yet."
            self.log.error(f"[TOOL RESULT] boundary_detector failed: {error_msg}")
            return json.dumps({"error": error_msg})

        maybe_err = self._ensure_boundary_model(model_path)
        if maybe_err:
            self.log.error(f"[TOOL RESULT] boundary_detector failed: {maybe_err}")
            return json.dumps({"error": maybe_err})

        text = self.state["binary"]["text_array"]
        base = int(self.state["binary"]["base_address"])
        self.log.debug(f"Running boundary detection on {len(text)} bytes, base=0x{base:08X}")

        result = self.boundaries.run(text, base)
        if not result.get("success"):
            error_msg = result.get("error", "Boundary detection failed")
            self.log.error(f"[TOOL RESULT] boundary_detector failed: {error_msg}")
            return json.dumps({"error": error_msg})

        funcs = result.get("functions", [])
        self.state["functions"] = funcs
        count = int(result.get("total_functions", len(funcs)))

        # Log detailed function boundaries
        self.log.info("="*60)
        self.log.info(f"[BOUNDARY DETECTOR RESULTS]")
        self.log.info(f"  Total functions detected: {count}")
        self.log.info("-"*60)

        # Log first 20 functions in detail, then summary for rest
        for i, func in enumerate(funcs[:20]):
            self.log.info(f"  Function #{i+1}:")
            self.log.info(f"    Type: {func.get('type', 'unknown')}")
            self.log.info(f"    Start Address: 0x{func.get('start_address', 0):08X}")
            self.log.info(f"    Start Offset: 0x{func.get('start_offset', 0):08X}")
            self.log.info(f"    End Offset: 0x{func.get('end_offset', 0):08X}")
            self.log.info(f"    Size: {func.get('size', 0)} bytes")

        if len(funcs) > 20:
            self.log.info(f"  ... and {len(funcs) - 20} more functions")

        self.log.info("="*60)

        # Dump full function list to debug log
        if self.verbose:
            self.log.debug("[FULL FUNCTION LIST]")
            self.log.debug(json.dumps(funcs, indent=2))

        examples = [hex(f.get("start_address", 0)) for f in funcs[:3]]
        success_msg = {
            "summary": f"Detected {count} functions.",
            "count": count,
            "examples": examples,
        }
        return json.dumps(success_msg)

    def _call_disassembler(self) -> str:
        self.log.info(f"[TOOL EXEC] disassembler")

        if not self.state.get("binary") or not self.state.get("functions"):
            error_msg = "No functions to disassemble."
            self.log.error(f"[TOOL RESULT] disassembler failed: {error_msg}")
            return json.dumps({"error": error_msg})

        text = self.state["binary"]["text_array"]
        base = int(self.state["binary"]["base_address"])
        arch = self.state["binary"]["architecture"]
        funcs = self.state["functions"]

        self.log.debug(f"Disassembling {len(funcs)} functions, base=0x{base:08X}, arch={arch}")
        result = self.disasm.run(text, funcs, base, arch)

        if not result.get("success"):
            error_msg = result.get("error", "Disassembly failed")
            self.log.error(f"[TOOL RESULT] disassembler failed: {error_msg}")
            return json.dumps({"error": error_msg})

        disassembled = result.get("disassembled_functions", [])
        self.state["disassembly"] = disassembled
        n = int(result.get("total_functions", len(disassembled)))

        # Log detailed disassembly results
        self.log.info("="*60)
        self.log.info(f"[DISASSEMBLER RESULTS]")
        self.log.info(f"  Total functions disassembled: {n}")
        self.log.info("-"*60)

        # Log first 5 functions' disassembly in detail
        for i, func_data in enumerate(disassembled[:5]):
            func_info = func_data.get("function", {})
            instructions = func_data.get("instructions", [])
            preview = func_data.get("disassembly_preview", "")

            self.log.info(f"  Function #{i+1}:")
            self.log.info(f"    Start Address: 0x{func_info.get('start_address', 0):08X}")
            self.log.info(f"    Size: {func_info.get('size', 0)} bytes")
            self.log.info(f"    Instruction Count: {func_data.get('instruction_count', 0)}")
            self.log.info(f"    First 20 instructions:")

            # Log the preview (already formatted)
            for line in preview.split('\n')[:20]:
                if line.strip():
                    self.log.info(f"      {line}")

        if len(disassembled) > 5:
            self.log.info(f"  ... and {len(disassembled) - 5} more functions")

        self.log.info("="*60)

        # Dump full disassembly to debug log (be careful with large files)
        if self.verbose and len(disassembled) <= 50:  # Only dump if not too many functions
            self.log.debug("[FULL DISASSEMBLY DUMP - First 50 functions]")
            for i, func_data in enumerate(disassembled[:50]):
                self.log.debug(f"Function {i+1}:")
                self.log.debug(json.dumps({
                    "function": func_data.get("function"),
                    "instruction_count": func_data.get("instruction_count"),
                    "first_10_instructions": func_data.get("instructions", [])[:10]
                }, indent=2))

        example_preview = None
        if disassembled:
            f0 = disassembled[0]
            addr = hex(f0["function"]["start_address"])
            preview = (f0.get("disassembly_preview") or "").strip()
            example_preview = {"addr": addr, "preview": preview[:200]}  # Truncate preview

        success_msg = {
            "summary": f"Disassembled {n} functions.",
            "total": n,
            "example": example_preview,
        }
        return json.dumps(success_msg)

    # ---------- Agent loop ----------

    def analyze(self, file_path: str, *, system_prompt: Optional[str] = None, max_iters: int = 8) -> str:
        import datetime
        import os

        messages = [
            {"role": "system", "content": system_prompt or "You are a concise binary analyst."},
            {"role": "user", "content": f"Analyze this binary and explain what it does: {file_path}"},
        ]
        path_hint = file_path

        self.log.info(f"[AGENT] Starting analysis of {file_path} (max iterations: {max_iters})")
        self.log.info(f"[AGENT] Binary file: {os.path.abspath(file_path) if os.path.exists(file_path) else 'File not found!'}")
        self.log.info(f"[AGENT] Timestamp: {datetime.datetime.now()}")

        for i in range(max_iters):
            self.log.info(f"[AGENT] === Iteration {i+1}/{max_iters} ===")

            # Log the full conversation history in verbose mode
            if self.verbose:
                self.log.debug(f"[CONVERSATION] Total messages: {len(messages)}")
                for idx, msg in enumerate(messages[-3:]):  # Last 3 messages
                    role = msg.get('role', 'unknown')
                    if role == 'function':
                        self.log.debug(f"  Message {len(messages)-3+idx}: {role} - {msg.get('name', 'unknown')}")
                    else:
                        content_preview = str(msg.get('content', ''))[:100]
                        self.log.debug(f"  Message {len(messages)-3+idx}: {role} - {content_preview}...")

            # Make the API call
            self.log.debug("[LLM] Sending request to model...")
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    functions=self.functions,
                    function_call="auto",
                )
                msg = resp.choices[0].message
            except Exception as e:
                self.log.error(f"[ERROR] API call failed: {e}")
                return f"Error: Failed to communicate with model: {e}"

            # Log the full model response in verbose mode
            if self.verbose:
                self.log.debug(f"[LLM RESPONSE] Content: {msg.content}")
                if hasattr(msg, 'function_call') and msg.function_call:
                    self.log.debug(f"[LLM RESPONSE] Function call: {msg.function_call.name}")
                    self.log.debug(f"[LLM RESPONSE] Function args: {msg.function_call.arguments}")

            # Log the model's reasoning/response
            if msg.content:
                # Log full content to file, truncated to console
                self.log.info(f"[LLM REASONING]\n{msg.content}")

            # Check if model wants to call a tool
            if getattr(msg, "function_call", None):
                name = msg.function_call.name
                raw = msg.function_call.arguments or "{}"

                self.log.info(f"[LLM DECISION] Model wants to call tool: {name}")
                self.log.debug(f"[LLM DECISION] Tool arguments: {raw}")

                try:
                    parsed = json.loads(raw)
                except Exception as e:
                    self.log.error(f"[ERROR] Failed to parse tool arguments: {e}")
                    parsed = {}

                # Auto-fill file_path for binary_loader if missing
                if name == bl.BinaryLoaderTool.name and "file_path" not in parsed:
                    self.log.debug(f"[AUTO-FILL] Adding file_path={path_hint} to binary_loader")
                    parsed["file_path"] = path_hint

                func, ArgsModel = self._tools.get(name, (None, None))
                if func is None:
                    content = json.dumps({"error": f"Unknown tool: {name}"})
                    self.log.error(f"[ERROR] Unknown tool requested: {name}")
                else:
                    kwargs, err = self._validate_args(parsed, ArgsModel)
                    if err:
                        content = json.dumps({"error": f"Invalid arguments: {err}"})
                        self.log.error(f"[ERROR] Invalid tool arguments: {err}")
                    else:
                        # Execute the tool
                        content = func(**kwargs)  # type: ignore[arg-type]

                # Add function result to messages
                messages.append({"role": "function", "name": name, "content": content})
                continue

            # Final answer received (no more tools)
            final = msg.content or ""
            self.log.info("[AGENT] === Final answer received ===")
            self.log.info(f"[FINAL ANSWER]\n{final}")
            self.log.info("[AGENT] Analysis complete")
            self.log.info("="*80)
            return final.strip()

        self.log.warning("[AGENT] Max iterations reached without final answer")
        self.log.warning("="*80)
        return "Stopped after max iterations without a final answer."
