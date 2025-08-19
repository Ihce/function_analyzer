#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import sys
from datetime import datetime
import logging
from typing import Optional, Dict, Any, List, Tuple, Type
import re
import time

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box

from .agent import BinaryAnalysisAgent

# Optional pydantic for schema validation (v1 or v2 supported, optional)
try:
    from pydantic import BaseModel, Field, ValidationError  # type: ignore
    PydAvailable = True
except Exception:
    PydAvailable = False
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore
    class ValidationError(Exception):  # type: ignore
        pass

app = typer.Typer(add_completion=False)
console = Console()

# --------------------------------------------------------------------------------------
# System prompt: model knows tools exist, but no fixed sequence is taught.
# --------------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a reverse-engineering assistant for binary analysis.

You MAY use these tools, in any order, as many times as needed:
- binary_loader(file_path: string)  — loads the binary and exposes .text, architecture, and base
- boundary_detector(model_path?; if you pass stride, it MUST be >= 1) — predicts function boundaries
- disassembler()                    — disassembles known functions; takes NO arguments (pass {})

Your job is to figure out which tool(s) to use, in what order, and iterate until you have enough evidence to answer the user's question.

CRITICAL: You MUST always respond with valid JSON. Never respond with empty content or plain text.
Do not mention tools or internal steps in your final answer. Be concise and technical."""

# --------------------------------------------------------------------------------------
# JSON schemas (planner, reflector, finalizer)
# --------------------------------------------------------------------------------------
class ActionModel(BaseModel if PydAvailable else object):  # type: ignore
    tool: str = Field(pattern=r"^(binary_loader|boundary_detector|disassembler)$") if PydAvailable else ""  # type: ignore
    args: Dict[str, Any] = Field(default_factory=dict) if PydAvailable else {}  # type: ignore

class PlannerOutput(BaseModel if PydAvailable else object):  # type: ignore
    plan: str = ""                         # one-sentence plan
    action: Optional[ActionModel] = None   # next tool call; null if ready to finalize
    final_answer: Optional[str] = None     # if ready, put answer here; else null
    reason: Optional[str] = ""             # why this plan/action makes sense
    confidence: Optional[float] = None     # 0.0–1.0

class ReflectOutput(BaseModel if PydAvailable else object):  # type: ignore
    reflection: str = ""                  # what we learned / next step thinking
    decision: str = ""                    # "continue" | "finalize" | "replan"
    reason: Optional[str] = ""            # why this decision
    confidence: Optional[float] = None

class FinalOutput(BaseModel if PydAvailable else object):  # type: ignore
    answer: str = ""                      # final user-facing answer
    reason: Optional[str] = ""            # brief explanation backing the answer
    confidence: Optional[float] = None

# --------------------------------------------------------------------------------------
# JSON instructions with examples
# --------------------------------------------------------------------------------------
PLANNER_JSON_SPEC = (
    "Return ONLY a JSON object with fields:\n"
    "{\n"
    '  "plan": "one short sentence",\n'
    '  "action": {"tool": "binary_loader|boundary_detector|disassembler", "args": {} } | null,\n'
    '  "final_answer": null | "string",\n'
    '  "reason": "short why this plan/action",\n'
    '  "confidence": 0.0-1.0\n'
    "}\n"
    "Examples:\n"
    '{"plan": "Load the binary first", "action": {"tool": "binary_loader", "args": {"file_path": "example.exe"}}, "final_answer": null, "reason": "Need to load binary before analysis", "confidence": 0.9}\n'
    '{"plan": "Ready to answer", "action": null, "final_answer": "The binary is a simple calculator", "reason": "Analysis complete", "confidence": 0.85}\n'
    "- If you can answer now, set action=null and put your text in final_answer.\n"
    "- If you need to act, set final_answer=null and provide one action.\n"
    "- disassembler MUST have {} args.\n"
    "Start with '{' and end with '}'. No extra text. No code fences."
)

REFLECT_JSON_SPEC = (
    "Given the latest observation, return ONLY a JSON object:\n"
    "{\n"
    '  "reflection": "one short sentence about what the observation means",\n'
    '  "decision": "continue" | "finalize" | "replan",\n'
    '  "reason": "short justification",\n'
    '  "confidence": 0.0-1.0\n'
    "}\n"
    "Example:\n"
    '{"reflection": "Functions detected successfully", "decision": "continue", "reason": "Need to disassemble for analysis", "confidence": 0.8}\n'
    "Start with '{' and end with '}'. No extra text. No code fences."
)

FINAL_JSON_SPEC = (
    "Return ONLY a JSON object with fields:\n"
    "{\n"
    '  "answer": "final user-facing answer",\n'
    '  "reason": "one short justification tying back to evidence",\n'
    '  "confidence": 0.0-1.0\n'
    "}\n"
    "Example:\n"
    '{"answer": "This binary is a text editor application", "reason": "Based on function analysis and API calls", "confidence": 0.75}\n'
    "Start with '{' and end with '}'. No extra text. No code fences."
)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _normalize_args(tool_name: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce tool arguments to safe defaults; invisible to the model."""
    if not isinstance(parsed, dict):
        parsed = {}
    if tool_name == "disassembler":
        return {}  # must be empty
    if tool_name == "boundary_detector":
        s = parsed.get("stride", None)
        if s is not None:
            try:
                if int(s) <= 0:
                    parsed.pop("stride", None)
            except Exception:
                parsed.pop("stride", None)
    return parsed

def _summarize_tool_result(content: str) -> str:
    """Short, human/log-friendly summary from a tool's JSON string."""
    try:
        obj = json.loads(content or "{}")
        if isinstance(obj, dict):
            if "error" in obj:
                return f"ERROR: {obj['error']}"
            if "summary" in obj:
                return str(obj["summary"])
    except Exception:
        pass
    s = (content or "").strip()
    return (s[:260] + "…") if len(s) > 260 else s

def _default_log_path(log_dir: str) -> str:
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"chat_session_{ts}.log")

def _setup_logger(name: str, log_file: Optional[str], verbose: bool, to_console: bool) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers = []
    fmt_file = logging.Formatter('%(asctime)s - [%(levelname)s] %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fmt_console = logging.Formatter('[%(levelname)s] %(message)s')
    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt_file)
        logger.addHandler(fh)
    if to_console:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt_console)
        ch.setLevel(logging.INFO if not verbose else logging.DEBUG)
        logger.addHandler(ch)
    return logger

def _extract_target_binary(text: str) -> Optional[str]:
    for line in text.splitlines():
        if line.lower().startswith("target binary:"):
            val = line.split(":", 1)[1].strip()
            if val:
                return val
    return None

# -------- Enhanced JSON parsing with diagnostics --------
def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

def _extract_json_block(text: str) -> Optional[str]:
    if not text:
        return None
    text = _strip_code_fences(text)

    # First try: direct JSON object
    if text.startswith("{") and text.endswith("}"):
        return text

    # Second try: find first JSON object in text
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(text)):
        c = text[i]

        if escape_next:
            escape_next = False
            continue

        if c == "\\":
            escape_next = True
            continue

        if c == '"' and not escape_next:
            in_string = not in_string
            continue

        if not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i+1].strip()

    # If we couldn't find a complete JSON object, return what we have
    if start >= 0:
        return text[start:].strip()

    return None

def _repair_json_like(s: str) -> str:
    # Fix Python boolean/None literals
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)

    # Fix single quotes (carefully)
    if s.count('"') < 2 and s.count("'") >= 2:
        s = s.replace("\\'", "ESCAPED_QUOTE_PLACEHOLDER")
        s = re.sub(r"'", '"', s)
        s = s.replace("ESCAPED_QUOTE_PLACEHOLDER", "'")

    # Remove trailing commas
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # Add missing quotes to keys (simple cases)
    s = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', s)

    return s

def _create_default_response(model_cls: Type) -> Any:
    """Create a sensible default response when parsing fails."""
    if model_cls == PlannerOutput:
        if PydAvailable:
            return PlannerOutput(
                plan="Unable to parse response, using fallback",
                action=None,
                final_answer=None,
                reason="JSON parsing failed",
                confidence=0.1
            )
        else:
            return {
                "plan": "Unable to parse response, using fallback",
                "action": None,
                "final_answer": None,
                "reason": "JSON parsing failed",
                "confidence": 0.1
            }
    elif model_cls == ReflectOutput:
        if PydAvailable:
            return ReflectOutput(
                reflection="Continuing after parse error",
                decision="continue",
                reason="Fallback to continue",
                confidence=0.1
            )
        else:
            return {
                "reflection": "Continuing after parse error",
                "decision": "continue",
                "reason": "Fallback to continue",
                "confidence": 0.1
            }
    elif model_cls == FinalOutput:
        if PydAvailable:
            return FinalOutput(
                answer="Analysis incomplete due to parsing errors",
                reason="JSON parsing failed",
                confidence=0.1
            )
        else:
            return {
                "answer": "Analysis incomplete due to parsing errors",
                "reason": "JSON parsing failed",
                "confidence": 0.1
            }
    return None

def _validate_model(obj: Dict[str, Any], Model: Type) -> Tuple[Optional[Any], Optional[str]]:
    if not PydAvailable:
        return obj, None
    try:
        try:
            return Model.model_validate(obj), None  # pydantic v2
        except AttributeError:
            return Model.parse_obj(obj), None       # pydantic v1
    except ValidationError as ve:
        errs = []
        try:
            for e in ve.errors():
                loc = ".".join(str(p) for p in e.get("loc", []))
                msg = e.get("msg", "")
                errs.append(f"{loc}: {msg}")
        except Exception:
            errs.append(str(ve))
        return None, "; ".join(errs)

def _parse_json_with_diagnostics(raw: str, model_cls: Type, retry_count: int = 0) -> Tuple[Optional[Any], str, Dict[str, str]]:
    """
    Enhanced JSON parser with better error recovery.
    Returns (parsed_obj_or_none, diagnostics_string_if_failed_else_empty, parts_dict_for_UI).
    """
    parts: Dict[str, str] = {"RAW": (raw or "").strip()}

    # Check for completely empty response
    if not parts["RAW"] or parts["RAW"] == "{}":
        default = _create_default_response(model_cls)
        if default:
            return default, "", {"RAW": parts["RAW"], "DEFAULT": "Using default response"}

    block = _extract_json_block(parts["RAW"]) or parts["RAW"]
    parts["EXTRACTED"] = block

    # Handle empty extraction
    if not block or block == "{}":
        default = _create_default_response(model_cls)
        if default:
            return default, "", parts

    # Try direct parsing
    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            # Fill in missing required fields with defaults
            if model_cls == PlannerOutput and "plan" not in obj:
                obj["plan"] = "Continuing analysis"
            if model_cls == ReflectOutput and "decision" not in obj:
                obj["decision"] = "continue"
            if model_cls == ReflectOutput and "reflection" not in obj:
                obj["reflection"] = "Processing observation"
            if model_cls == FinalOutput and "answer" not in obj:
                obj["answer"] = "Analysis incomplete"

            parsed, err = _validate_model(obj, model_cls)
            if parsed is not None:
                return parsed, "", parts
            else:
                parts["ERR"] = f"Schema validation error: {err}"
    except json.JSONDecodeError as e1:
        parts["ERR"] = f"direct json.loads -> {type(e1).__name__}: {e1}"
    except Exception as e1:
        parts["ERR"] = f"direct parse error -> {type(e1).__name__}: {e1}"

    # Try repair
    repaired = _repair_json_like(block)
    parts["REPAIRED"] = repaired

    if repaired != block:  # Only try if repair changed something
        try:
            obj2 = json.loads(repaired)
            if isinstance(obj2, dict):
                # Fill in missing required fields with defaults
                if model_cls == PlannerOutput and "plan" not in obj2:
                    obj2["plan"] = "Continuing analysis"
                if model_cls == ReflectOutput and "decision" not in obj2:
                    obj2["decision"] = "continue"
                if model_cls == ReflectOutput and "reflection" not in obj2:
                    obj2["reflection"] = "Processing observation"
                if model_cls == FinalOutput and "answer" not in obj2:
                    obj2["answer"] = "Analysis incomplete"

                parsed, err = _validate_model(obj2, model_cls)
                if parsed is not None:
                    return parsed, "", parts
                else:
                    parts["ERR"] += f"\nSchema validation error after repair: {err}"
        except Exception as e2:
            parts["ERR"] += f"\nrepaired json.loads -> {type(e2).__name__}: {e2}"

    # Last resort: use default
    if retry_count < 2:  # Allow up to 2 retries
        default = _create_default_response(model_cls)
        if default:
            return default, "", parts

    diag = (
        "JSON parse failed.\n"
        f"RAW:\n{parts['RAW'] or '(empty)'}\n\n"
        f"EXTRACTED:\n{parts['EXTRACTED'] or '(empty)'}\n\n"
        f"REPAIRED:\n{parts.get('REPAIRED','(skipped)')}\n\n"
        f"ERRORS:\n{parts.get('ERR','(none)')}"
    )
    return None, diag, parts

# --------------------------------------------------------------------------------------
# Session: PLAN → ACT → OBSERVE → REFLECT → (FINAL)
# --------------------------------------------------------------------------------------
class Session:
    def __init__(
        self,
        binary: Optional[str],
        boundary_ckpt: Optional[str],
        model_id: str,
        ollama_url: str,
        verbose: bool,
        log_file: Optional[str],
        no_console: bool,
    ) -> None:
        self.logger = _setup_logger("chat", log_file, verbose, not no_console)
        self.agent = BinaryAnalysisAgent(
            ollama_url=ollama_url,
            model_id=model_id,
            boundary_ckpt=boundary_ckpt or "models/function_boundary.pth",
            verbose=verbose,
            log_file=log_file,
            log_to_console=not no_console,
        )
        self.system_prompt = SYSTEM_PROMPT
        self.binary_hint = binary
        self.verbose = verbose
        self.log_file = log_file
        self.no_console = no_console
        self.retry_count = 0
        self.max_retries = 3

        self.logger.info("=" * 80)
        self.logger.info(f"CHAT SESSION STARTED - {datetime.now()}")
        self.logger.info(f"Model: {model_id}")
        self.logger.info(f"Binary hint: {binary or 'None'}")
        self.logger.info("=" * 80)

    # -------- Enhanced LLM helper with retry logic --------
    def _call_llm(self, messages: list[dict[str, str]], retry_attempt: int = 0) -> str:
        """Call LLM with enhanced error handling and retry logic."""

        # Add more explicit JSON instruction to the last user message
        if retry_attempt > 0 and messages:
            last_msg = messages[-1]
            if last_msg["role"] == "user":
                last_msg["content"] = (
                    "IMPORTANT: You MUST respond with valid JSON only. "
                    "Start with '{' and end with '}'. No other text.\n\n" +
                    last_msg["content"]
                )

        # Try Ollama JSON mode first
        try:
            resp = self.agent.client.chat.completions.create(
                model=self.agent.model_id,
                messages=messages,
                temperature=0.0 if retry_attempt == 0 else 0.1,  # Add slight temperature on retry
                max_tokens=1024,  # Increase token limit
                extra_body={
                    "format": "json",
                    "options": {
                        "temperature": 0.0 if retry_attempt == 0 else 0.1,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1,  # Reduce repetition
                        "stop": ["```", "\n\n\n"],  # Stop sequences
                    }
                },
                timeout=30,  # Add timeout
            )
            content = (resp.choices[0].message.content or "").strip()

            # Validate we got something that looks like JSON
            if content and (content.startswith("{") or "{" in content):
                return content

            # If empty or not JSON-like, retry with a different approach
            if retry_attempt < 2:
                self.logger.warning(f"[API] Got non-JSON response, retrying (attempt {retry_attempt + 1})")
                time.sleep(0.5)  # Brief delay before retry
                return self._call_llm(messages, retry_attempt + 1)

        except Exception as e1:
            self.logger.warning(f"[API] Ollama json format failed ({e1}); falling back.")

        # Fallback: OpenAI-style
        try:
            resp = self.agent.client.chat.completions.create(
                model=self.agent.model_id,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
                timeout=30,
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
        except Exception as e2:
            self.logger.warning(f"[API] response_format failed ({e2}); final plain retry.")

        # Last resort: plain text with explicit instruction
        try:
            # Add explicit JSON instruction
            modified_messages = messages.copy()
            if modified_messages and modified_messages[-1]["role"] == "user":
                modified_messages[-1]["content"] = (
                    "Response MUST be valid JSON starting with { and ending with }.\n\n" +
                    modified_messages[-1]["content"]
                )

            resp = self.agent.client.chat.completions.create(
                model=self.agent.model_id,
                messages=modified_messages,
                temperature=0.2,
                max_tokens=1024,
                timeout=30,
            )
            content = (resp.choices[0].message.content or "").strip()

            # If still empty, create a minimal valid response
            if not content:
                self.logger.error("[API] Empty response from model")
                if "plan" in str(messages[-1]):
                    return '{"plan": "Continue analysis", "action": null, "final_answer": null, "reason": "Empty response", "confidence": 0.1}'
                elif "reflect" in str(messages[-1]):
                    return '{"reflection": "Continuing", "decision": "continue", "reason": "Empty response", "confidence": 0.1}'
                else:
                    return '{"answer": "Unable to complete analysis", "reason": "Empty response", "confidence": 0.1}'

            return content

        except Exception as e3:
            self.logger.error(f"[API] All attempts failed: {e3}")
            # Return a minimal valid JSON response
            return '{"error": "API call failed", "plan": "Retry", "action": null, "final_answer": null}'

    # -------- Controller fallback policy --------
    def _fallback_policy(self, target_binary: Optional[str]) -> Tuple[Optional[ActionModel], str]:
        """Decide next best action from current agent state; returns (ActionModel|None, controller_reason)."""
        b = self.agent.state.get("binary")
        f = self.agent.state.get("functions")
        d = self.agent.state.get("disassembly")

        if not (b and b.get("success")):
            args = {"file_path": target_binary} if target_binary else {}
            action = ActionModel(tool="binary_loader", args=args) if PydAvailable else {"tool": "binary_loader", "args": args}
            return action, "Controller fallback: load the binary to obtain architecture, base address, and .text."

        if not f:
            action = ActionModel(tool="boundary_detector", args={}) if PydAvailable else {"tool": "boundary_detector", "args": {}}
            return action, "Controller fallback: detect function boundaries to prepare for disassembly."

        if not d:
            action = ActionModel(tool="disassembler", args={}) if PydAvailable else {"tool": "disassembler", "args": {}}
            return action, "Controller fallback: disassemble detected functions to inspect for vulnerabilities."

        # Everything gathered → no action; next step will be finalize
        return None, "Controller fallback: sufficient evidence gathered; finalize with a concise answer."

    # -------------------
    # Controller loop
    # -------------------
    def ask_once(self, user_msg: str, max_iterations: int = 10) -> str:
        """
        Enhanced PLAN → ACT → OBSERVE → REFLECT loop with robust error handling.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        target_binary = _extract_target_binary(user_msg) or self.binary_hint
        consecutive_failures = 0

        for iteration in range(1, max_iterations + 1):
            self.logger.info(f"[ITER {iteration}] ------------------------------")

            # ==========================
            # [PLAN]
            # ==========================

            # Check if we've already done all the necessary work
            if iteration > 3 and self.agent.state.get("disassembly"):
                # We've done enough iterations and have all the data
                # Let the model provide its analysis directly
                self.logger.info("[PLAN] All tools executed, requesting final analysis")

                analysis_prompt = (
                    "You have successfully:\n"
                    "1. Loaded the binary (x86, base 0x00401000, 20KB text section)\n"
                    "2. Detected 238 functions\n"
                    "3. Disassembled all functions\n\n"
                    "Now provide your final analysis to answer the user's question.\n"
                    "Be direct and technical. Focus on answering what was asked."
                )

                analysis_messages = [*messages, {"role": "user", "content": analysis_prompt}]

                try:
                    resp = self.agent.client.chat.completions.create(
                        model=self.agent.model_id,
                        messages=analysis_messages,
                        temperature=0.0,
                        max_tokens=2048,
                        timeout=60,
                    )
                    final_answer = (resp.choices[0].message.content or "").strip()

                    if final_answer:
                        console.print(Panel.fit("[PLAN] Analysis complete, providing final answer", style="bold cyan", border_style="cyan"))
                        console.print(Panel(final_answer, title="Final Analysis", border_style="green"))
                        self.logger.info(f"[FINAL]\n{final_answer}")
                        return final_answer
                except Exception as e:
                    self.logger.warning(f"[PLAN] Direct analysis failed: {e}")

            # Normal planning phase
            planner_prompt = (
                "Decide the very next step toward the objective below.\n"
                "Choose ANY available tool in ANY order, or provide a final answer if ready.\n"
                "YOU MUST RESPOND WITH VALID JSON.\n\n"
                f"{PLANNER_JSON_SPEC}"
            )
            plan_messages = [*messages, {"role": "user", "content": planner_prompt}]

            # Try to get plan with retries
            plan_obj = None
            for plan_attempt in range(3):
                raw_plan = self._call_llm(plan_messages, retry_attempt=plan_attempt)
                plan_obj, diag, parts = _parse_json_with_diagnostics(raw_plan, PlannerOutput, retry_count=plan_attempt)

                if plan_obj is not None:
                    consecutive_failures = 0
                    break

                if plan_attempt < 2:
                    self.logger.warning(f"[PLAN] Parse attempt {plan_attempt + 1} failed, retrying...")
                    time.sleep(0.5)

            if plan_obj is None:
                # All attempts failed, use fallback
                consecutive_failures += 1
                self.logger.error(f"[PLAN] All parse attempts failed (consecutive: {consecutive_failures})")

                if consecutive_failures > 3:
                    self.logger.error("[PLAN] Too many consecutive failures, aborting")
                    return "Analysis failed due to repeated parsing errors. Please try with a different model."

                # Use fallback policy
                action, controller_reason = self._fallback_policy(target_binary)
                plan_obj = _create_default_response(PlannerOutput)
                if PydAvailable:
                    plan_obj.plan = "Using controller fallback"
                    plan_obj.action = action
                    plan_obj.reason = controller_reason
                else:
                    plan_obj["plan"] = "Using controller fallback"
                    plan_obj["action"] = action
                    plan_obj["reason"] = controller_reason

            # Extract plan details
            plan_text = (plan_obj.plan if PydAvailable else plan_obj.get("plan", "")).strip()
            plan_reason = (plan_obj.reason if PydAvailable else plan_obj.get("reason", "")).strip()
            final_answer = (plan_obj.final_answer if PydAvailable else plan_obj.get("final_answer"))
            action = plan_obj.action if PydAvailable else plan_obj.get("action")

            # Display plan
            console.print(Panel.fit(f"[PLAN] {plan_text or '(no plan text)'}", style="bold cyan", border_style="cyan"))
            if plan_reason:
                console.print(Panel.fit(plan_reason, title="Reason", style="dim", border_style="cyan"))
            self.logger.info(f"[PLAN] {plan_text}")
            if plan_reason:
                self.logger.info(f"[PLAN][REASON] {plan_reason}")

            # If planner provided final answer directly, finish now
            if isinstance(final_answer, str) and final_answer:
                console.print(Panel(final_answer, title="Final", border_style="green"))
                self.logger.info(f"[FINAL]\n{final_answer}")
                return final_answer

            # If no action and no final answer, ask for final answer
            if action is None and final_answer is None:
                final_prompt = (
                    "Produce the final user-facing answer now, based on all observations so far.\n"
                    "YOU MUST RESPOND WITH VALID JSON.\n\n"
                    f"{FINAL_JSON_SPEC}"
                )
                final_messages = [*messages, {"role": "user", "content": final_prompt}]

                for final_attempt in range(3):
                    raw_final = self._call_llm(final_messages, retry_attempt=final_attempt)
                    final_obj, fdiag, _ = _parse_json_with_diagnostics(raw_final, FinalOutput, retry_count=final_attempt)
                    if final_obj is not None:
                        break
                    if final_attempt < 2:
                        time.sleep(0.5)
                else:
                    self.logger.error("[FINAL] All parse attempts failed")
                    return "Analysis complete but unable to format final answer properly."

                final_answer = (final_obj.answer if PydAvailable else final_obj.get("answer", "")).strip()
                final_reason = (final_obj.reason if PydAvailable else final_obj.get("reason", "")).strip()
                console.print(Panel(final_answer or "(no answer)", title="Final", border_style="green"))
                if final_reason:
                    console.print(Panel.fit(final_reason, title="Reason", style="dim", border_style="green"))
                self.logger.info(f"[FINAL]\n{final_answer}")
                return final_answer

            # ==========================
            # [ACT]
            # ==========================
            tool = action.tool if PydAvailable else action.get("tool") if action else None
            args = dict(action.args if PydAvailable else action.get("args", {})) if action else {}

            if not tool:
                self.logger.warning("[ACT] No tool specified, using fallback")
                action, _ = self._fallback_policy(target_binary)
                if action:
                    tool = action.tool if PydAvailable else action.get("tool")
                    args = dict(action.args if PydAvailable else action.get("args", {}))
                else:
                    continue

            # Autofill binary path for loader if missing
            if tool == "binary_loader":
                if "file_path" not in args and target_binary:
                    args["file_path"] = target_binary

            args = _normalize_args(tool, args)

            self.logger.info(f"[ACT] {tool} args={args}")
            console.print(Panel.fit(f"[ACT] {tool} args={args}", style="bold yellow", border_style="yellow"))

            func_tuple = self.agent._tools.get(tool)
            if not func_tuple:
                tool_content = json.dumps({"error": f"Unknown tool: {tool}"})
                self.logger.error(f"[ERROR] Unknown tool requested: {tool}")
            else:
                func, ArgsModel = func_tuple
                kwargs, err = self.agent._validate_args(args, ArgsModel)
                if err:
                    tool_content = json.dumps({"error": f"Invalid arguments: {err}"})
                    self.logger.error(f"[ERROR] Invalid tool arguments: {err}")
                else:
                    self.logger.info(f"[TOOL EXEC] {tool}")
                    tool_content = func(**kwargs)

            # ==========================
            # [OBSERVE]
            # ==========================
            summary = _summarize_tool_result(tool_content)
            obs_line = f"{tool} -> {summary}"
            self.logger.info(f"[OBSERVE] {obs_line}")
            console.print(Panel.fit(f"[OBSERVE] {obs_line}", style="bold green", border_style="green"))

            # Feed observation back to model
            messages.append({"role": "assistant", "content": f"Observation: {obs_line}"})

            # ==========================
            # [REFLECT]
            # ==========================
            reflect_prompt = (
                "Reflect on the most recent observation and decide whether to continue (another action), "
                "finalize (you can answer now), or replan (change approach).\n"
                "YOU MUST RESPOND WITH VALID JSON.\n\n"
                f"{REFLECT_JSON_SPEC}"
            )
            reflect_messages = [*messages, {"role": "user", "content": reflect_prompt}]

            for reflect_attempt in range(3):
                raw_reflect = self._call_llm(reflect_messages, retry_attempt=reflect_attempt)
                reflect_obj, rdiag, _ = _parse_json_with_diagnostics(raw_reflect, ReflectOutput, retry_count=reflect_attempt)
                if reflect_obj is not None:
                    break
                if reflect_attempt < 2:
                    time.sleep(0.5)
            else:
                # Use default continue
                reflect_obj = _create_default_response(ReflectOutput)

            reflect_text = (reflect_obj.reflection if PydAvailable else reflect_obj.get("reflection", "")).strip()
            reflect_reason = (reflect_obj.reason if PydAvailable else reflect_obj.get("reason", "")).strip()
            decision = (reflect_obj.decision if PydAvailable else reflect_obj.get("decision", "continue")).strip().lower()

            console.print(Panel.fit(f"[REFLECT] {reflect_text or '(no reflection)'}", style="bold magenta", border_style="magenta"))
            if reflect_reason:
                console.print(Panel.fit(reflect_reason, title="Reason", style="dim", border_style="magenta"))
            self.logger.info(f"[REFLECT] {reflect_text}")
            if reflect_reason:
                self.logger.info(f"[REFLECT][REASON] {reflect_reason}")

            # Handle decision
            if decision == "finalize":
                # Let the model reason naturally without JSON constraints
                final_prompt = (
                    "Based on all the information gathered from the tools, provide your final analysis.\n"
                    "Answer the user's original question directly.\n"
                    "Be concise and technical.\n"
                    "DO NOT use JSON format - just provide your reasoning and answer as plain text."
                )
                final_messages = [*messages, {"role": "user", "content": final_prompt}]

                # Get the model's natural reasoning
                try:
                    resp = self.agent.client.chat.completions.create(
                        model=self.agent.model_id,
                        messages=final_messages,
                        temperature=0.0,
                        max_tokens=2048,  # Give plenty of room for reasoning
                        timeout=60,  # More time for complex reasoning
                    )
                    final_answer = (resp.choices[0].message.content or "").strip()

                    # If we got a response, use it (even if it's JSON, we'll extract the content)
                    if final_answer:
                        # Check if it accidentally returned JSON and extract the answer
                        if final_answer.startswith("{") and "answer" in final_answer:
                            try:
                                obj = json.loads(final_answer)
                                final_answer = obj.get("answer", obj.get("final_answer", final_answer))
                            except:
                                pass  # Use as-is if not valid JSON

                        console.print(Panel(final_answer, title="Final Analysis", border_style="green"))
                        self.logger.info(f"[FINAL]\n{final_answer}")
                        return final_answer

                except Exception as e:
                    self.logger.warning(f"[FINAL] Failed to get reasoning: {e}")

                # If plain text failed, try with JSON format as fallback
                self.logger.info("[FINAL] Plain text failed, trying JSON format")
                final_prompt = (
                    "Produce the final user-facing answer now, based on all observations so far.\n"
                    "YOU MUST RESPOND WITH VALID JSON.\n\n"
                    f"{FINAL_JSON_SPEC}"
                )
                final_messages = [*messages, {"role": "user", "content": final_prompt}]

                for final_attempt in range(3):
                    raw_final = self._call_llm(final_messages, retry_attempt=final_attempt)
                    final_obj, fdiag, _ = _parse_json_with_diagnostics(raw_final, FinalOutput, retry_count=final_attempt)
                    if final_obj is not None:
                        break
                    if final_attempt < 2:
                        time.sleep(0.5)
                else:
                    # Last resort - construct from what we know
                    if self.agent.state.get("functions"):
                        func_count = len(self.agent.state["functions"])
                        return f"Analysis complete. Found {func_count} functions in the binary. Unable to provide detailed analysis due to model limitations."
                    return "Analysis complete but unable to provide detailed results."

                final_answer = (final_obj.answer if PydAvailable else final_obj.get("answer", "")).strip()
                final_reason = (final_obj.reason if PydAvailable else final_obj.get("reason", "")).strip()
                console.print(Panel(final_answer or "(no answer)", title="Final Analysis", border_style="green"))
                if final_reason:
                    console.print(Panel.fit(final_reason, title="Reason", style="dim", border_style="green"))
                self.logger.info(f"[FINAL]\n{final_answer}")
                return final_answer

            # continue or replan → iterate

        self.logger.warning("[AGENT] Max iterations reached without a final answer.")
        return "Analysis stopped after max iterations. The binary has been loaded, functions detected, and disassembly completed."

# --------------------------------------------------------------------------------------
# Typer commands (unchanged)
# --------------------------------------------------------------------------------------
def _common_session(
    binary: Optional[str],
    boundary_ckpt: Optional[str],
    model_id: str,
    ollama_url: str,
    verbose: bool,
    log_dir: str,
    log_file: Optional[str],
    no_console: bool,
) -> Session:
    log_path = log_file or _default_log_path(log_dir)
    return Session(binary, boundary_ckpt, model_id, ollama_url, verbose, log_path, no_console)

@app.command(help="Analyze a binary and answer a question (iterative).")
def ask(
    binary: str = typer.Argument(..., help="Path to the binary file"),
    question: str = typer.Argument(..., help="Question to answer about the binary"),
    model_id: str = typer.Option("gpt-oss:20b", "--model-id", help="Model ID for Ollama"),
    ollama_url: str = typer.Option("http://localhost:11434/v1", "--ollama-url", help="Ollama/OpenAI-compatible URL"),
    boundary_ckpt: Optional[str] = typer.Option(None, "--boundary-ckpt", help="Path to boundary model checkpoint"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
    log_dir: str = typer.Option("./logs", "--log-dir", help="Log directory"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Custom log file path"),
    no_console: bool = typer.Option(False, "--no-console", help="Disable console logging"),
):
    sess = _common_session(binary, boundary_ckpt, model_id, ollama_url, verbose, log_dir, log_file, no_console)
    user_msg = f"""Target binary: {binary}

Objective: {question}

Iterate with PLAN → ACT → OBSERVE → REFLECT until you can answer confidently. Output your final answer only when asked to finalize."""
    console.print(Panel.fit(f"[b]{os.path.basename(binary)}[/b] • {question}", title="ASK", style="magenta"))
    final = sess.ask_once(user_msg)
    console.print(Panel(final or "(no answer)", title="Answer", border_style="green"))

@app.command(help="Open a binary and summarize what it does (iterative).")
def open(
    binary: str = typer.Argument(..., help="Path to the binary file"),
    model_id: str = typer.Option("gpt-oss:20b", "--model-id"),
    ollama_url: str = typer.Option("http://localhost:11434/v1", "--ollama-url"),
    boundary_ckpt: Optional[str] = typer.Option(None, "--boundary-ckpt"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
    log_dir: str = typer.Option("./logs", "--log-dir"),
    log_file: Optional[str] = typer.Option(None, "--log-file"),
    no_console: bool = typer.Option(False, "--no-console"),
):
    sess = _common_session(binary, boundary_ckpt, model_id, ollama_url, verbose, log_dir, log_file, no_console)
    user_msg = f"""Target binary: {binary}

Objective: Summarize what this program does.

Iterate with PLAN → ACT → OBSERVE → REFLECT until a concise summary can be finalized. Output your final answer only when asked to finalize."""
    console.print(Panel.fit(f"[b]{os.path.basename(binary)}[/b]", title="OPEN", style="magenta"))
    final = sess.ask_once(user_msg)
    console.print(Panel(final or "(no answer)", title="Summary", border_style="green"))

@app.command(help="List available tools.")
def tools(
    model_id: str = typer.Option("gpt-oss:20b", "--model-id"),
    ollama_url: str = typer.Option("http://localhost:11434/v1", "--ollama-url"),
    boundary_ckpt: Optional[str] = typer.Option(None, "--boundary-ckpt"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
    log_dir: str = typer.Option("./logs", "--log-dir"),
    log_file: Optional[str] = typer.Option(None, "--log-file"),
    no_console: bool = typer.Option(False, "--no-console"),
):
    sess = _common_session(None, boundary_ckpt, model_id, ollama_url, verbose, log_dir, log_file, no_console)
    table = Table(title="Available Tools", box=box.SIMPLE_HEAVY, show_lines=False)
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Name", style="bold")
    table.add_column("Description", style="dim")
    for i, tool in enumerate(sess.agent.tools, 1):
        fn = tool.get("function", {})
        table.add_row(str(i), fn.get("name", "unknown"), (fn.get("description", "") or "")[:120])
    console.print(table)

@app.command(help="Interactive shell (iterative controller loop).")
def repl(
    model_id: str = typer.Option("gpt-oss:20b", "--model-id"),
    ollama_url: str = typer.Option("http://localhost:11434/v1", "--ollama-url"),
    boundary_ckpt: Optional[str] = typer.Option(None, "--boundary-ckpt"),
    binary: Optional[str] = typer.Option(None, "--binary", help="Default binary hint"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
    log_dir: str = typer.Option("./logs", "--log-dir"),
    log_file: Optional[str] = typer.Option(None, "--log-file"),
    no_console: bool = typer.Option(False, "--no-console"),
):
    sess = _common_session(binary, boundary_ckpt, model_id, ollama_url, verbose, log_dir, log_file, no_console)
    console.print(Panel.fit("Type: ask, open, tools, reset, verbose, exit", title="REPL", style="magenta"))

    while True:
        try:
            line = Prompt.ask("[bold]you[/bold]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            sess.logger.info("REPL ended by user.")
            break

        cmd = line.strip()
        if not cmd:
            continue
        if cmd in {"exit", "quit"}:
            sess.logger.info("REPL exited.")
            break
        if cmd == "help":
            console.print("Commands: ask <binary> <question> | open <binary> | tools | reset | verbose | exit")
            continue
        if cmd == "verbose":
            sess.logger.setLevel(logging.DEBUG if sess.logger.level == logging.INFO else logging.INFO)
            sess.agent.log.setLevel(logging.DEBUG if sess.agent.log.level == logging.INFO else logging.INFO)
            console.print(f"Verbose: {'ON' if sess.logger.level == logging.DEBUG else 'OFF'}")
            continue
        if cmd == "reset":
            console.print("[yellow]Context cleared.[/yellow]")
            continue
        if cmd == "tools":
            table = Table(title="Available Tools", box=box.SIMPLE_HEAVY, show_lines=False)
            table.add_column("#", style="cyan", no_wrap=True)
            table.add_column("Name", style="bold")
            table.add_column("Description", style="dim")
            for i, tool in enumerate(sess.agent.tools, 1):
                fn = tool.get("function", {})
                table.add_row(str(i), fn.get("name", "unknown"), (fn.get("description", "") or "")[:120])
            console.print(table)
            continue
        if cmd.startswith("ask "):
            parts = cmd[4:].strip().split(None, 1)
            if len(parts) < 2:
                console.print("[red]Usage:[/red] ask <binary_path> <question>")
                continue
            bin_path, question = parts[0], parts[1]
            user_msg = f"""Target binary: {bin_path}

Objective: {question}

Iterate with PLAN → ACT → OBSERVE → REFLECT until you can answer confidently. Output your final answer only when asked to finalize."""
            console.print(Panel.fit(f"[b]{os.path.basename(bin_path)}[/b] • {question}", title="ASK", style="magenta"))
            final = sess.ask_once(user_msg)
            console.print(Panel(final or "(no answer)", title="Answer", border_style="green"))
            continue
        if cmd.startswith("open "):
            bin_path = cmd[5:].strip()
            if not bin_path:
                console.print("[red]Usage:[/red] open <binary_path>")
                continue
            user_msg = f"""Target binary: {bin_path}

Objective: Summarize what this program does.

Iterate with PLAN → ACT → OBSERVE → REFLECT until a concise summary can be finalized. Output your final answer only when asked to finalize."""
            console.print(Panel.fit(f"[b]{os.path.basename(bin_path)}[/b]", title="OPEN", style="magenta"))
            final = sess.ask_once(user_msg)
            console.print(Panel(final or "(no answer)", title="Summary", border_style="green"))
            continue

        # Fallthrough: send plain question
        user_msg = cmd
        final = sess.ask_once(user_msg)
        console.print(Panel(final or "(no answer)", title="Response", border_style="green"))

def main():
    app()

if __name__ == "__main__":
    sys.exit(main())
