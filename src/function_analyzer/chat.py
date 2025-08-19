#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import sys
from datetime import datetime
import logging
from typing import Optional, Dict, Any, List, Tuple, Type
import re

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

Your job is to figure out which tool(s) to use, in what order, and iterate until you have enough evidence to answer the user’s question.
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
# JSON instructions
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
    "Start with '{' and end with '}'. No extra text. No code fences."
)

FINAL_JSON_SPEC = (
    "Return ONLY a JSON object with fields:\n"
    "{\n"
    '  "answer": "final user-facing answer",\n'
    '  "reason": "one short justification tying back to evidence",\n'
    '  "confidence": 0.0-1.0\n'
    "}\n"
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

# -------- JSON parsing with diagnostics --------
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
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1].strip()
    return None

def _repair_json_like(s: str) -> str:
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    if s.count('"') < 2 and s.count("'") >= 2:
        s = s.replace("\\'", "'")
        s = re.sub(r"'", '"', s)
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s

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

def _parse_json_with_diagnostics(raw: str, model_cls: Type) -> Tuple[Optional[Any], str, Dict[str, str]]:
    """
    Returns (parsed_obj_or_none, diagnostics_string_if_failed_else_empty, parts_dict_for_UI).
    parts = {"RAW":..., "EXTRACTED":..., "REPAIRED":..., "ERR":...}
    """
    parts: Dict[str, str] = {"RAW": (raw or "").strip()}
    block = _extract_json_block(parts["RAW"]) or parts["RAW"]
    parts["EXTRACTED"] = block

    # Try direct
    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            parsed, err = _validate_model(obj, model_cls)
            if parsed is not None:
                return parsed, "", parts
            else:
                parts["ERR"] = f"Schema validation error: {err}"
                return None, f"Schema validation error: {err}", parts
    except Exception as e1:
        parts["ERR"] = f"direct json.loads -> {type(e1).__name__}: {e1}"

    repaired = _repair_json_like(block)
    parts["REPAIRED"] = repaired
    try:
        obj2 = json.loads(repaired)
        if isinstance(obj2, dict):
            parsed, err = _validate_model(obj2, model_cls)
            if parsed is not None:
                return parsed, "", parts
            else:
                parts["ERR"] = f"Schema validation error after repair: {err}"
                return None, f"Schema validation error after repair: {err}", parts
    except Exception as e2:
        parts["ERR"] += f"\nrepaired json.loads -> {type(e2).__name__}: {e2}"

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

        self.logger.info("=" * 80)
        self.logger.info(f"CHAT SESSION STARTED - {datetime.now()}")
        self.logger.info(f"Model: {model_id}")
        self.logger.info(f"Binary hint: {binary or 'None'}")
        self.logger.info("=" * 80)

    # -------- LLM helper (tries JSON mode; falls back) --------
    def _call_llm(self, messages: list[dict[str, str]]) -> str:
        # Try Ollama JSON mode first (many OSS backends only honor this)
        try:
            resp = self.agent.client.chat.completions.create(
                model=self.agent.model_id,
                messages=messages,
                temperature=0.0,
                max_tokens=512,  # prevent silent truncation
                extra_body={
                    "format": "json",          # <-- Ollama JSON enforcement
                    "options": {
                        "temperature": 0.0,
                        "top_p": 0.9,
                        # "num_ctx": 8192,     # set if your model supports larger context
                    }
                },
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
        except Exception as e1:
            self.logger.warning(f"[API] Ollama json format path failed ({e1}); falling back.")

        # Fallback: try OpenAI-style json_object (ignored by many shims)
        try:
            resp = self.agent.client.chat.completions.create(
                model=self.agent.model_id,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e2:
            self.logger.warning(f"[API] response_format path failed ({e2}); final plain retry.")

        # Last resort: plain text (we’ll still parse/repair)
        resp = self.agent.client.chat.completions.create(
            model=self.agent.model_id,
            messages=messages,
            temperature=0.1,     # a touch of entropy helps some OSS models avoid empties
            max_tokens=512,
        )
        return (resp.choices[0].message.content or "").strip()


    # -------- Controller fallback policy (used only when planner JSON fails) --------
    def _fallback_policy(self, target_binary: Optional[str]) -> Tuple[Optional[ActionModel], str]:
        """Decide next best action from current agent state; returns (ActionModel|None, controller_reason)."""
        b = self.agent.state.get("binary")
        f = self.agent.state.get("functions")
        d = self.agent.state.get("disassembly")

        if not (b and b.get("success")):
            args = {"file_path": target_binary} if target_binary else {}
            return ActionModel(tool="binary_loader", args=args) if PydAvailable else {"tool": "binary_loader", "args": args}, \
                   "Controller fallback: load the binary to obtain architecture, base address, and .text."

        if not f:
            return ActionModel(tool="boundary_detector", args={}) if PydAvailable else {"tool": "boundary_detector", "args": {}}, \
                   "Controller fallback: detect function boundaries to prepare for disassembly."

        if not d:
            return ActionModel(tool="disassembler", args={}) if PydAvailable else {"tool": "disassembler", "args": {}}, \
                   "Controller fallback: disassemble detected functions to inspect for vulnerabilities."

        # Everything gathered → no action; next step will be finalize
        return None, "Controller fallback: sufficient evidence gathered; finalize with a concise answer."

    # -------------------
    # Controller loop
    # -------------------
    def ask_once(self, user_msg: str, max_iterations: int = 10) -> str:
        """
        Explicit PLAN → ACT → OBSERVE → REFLECT loop, strict JSON I/O, Reason panels,
        robust diagnostics, and a controller fallback when planner misbehaves.
        """
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        target_binary = _extract_target_binary(user_msg) or self.binary_hint

        for iteration in range(1, max_iterations + 1):
            self.logger.info(f"[ITER {iteration}] ------------------------------")

            # ==========================
            # [PLAN]
            # ==========================
            planner_prompt = (
                "Decide the very next step toward the objective below.\n"
                "Choose ANY available tool in ANY order, or provide a final answer if ready.\n\n"
                f"{PLANNER_JSON_SPEC}"
            )
            plan_messages = [*messages, {"role": "user", "content": planner_prompt}]
            raw_plan = self._call_llm(plan_messages)
            plan_obj, diag, parts = _parse_json_with_diagnostics(raw_plan, PlannerOutput)

            # Retry once with STRICT JSON if needed
            if plan_obj is None:
                self.logger.warning("[PLAN] Malformed JSON; retrying with STRICT instruction.")
                strict = [*plan_messages, {"role": "user", "content": "STRICT JSON ONLY. Repeat exactly in the specified JSON schema."}]
                raw_plan2 = self._call_llm(strict)
                plan_obj, diag2, parts2 = _parse_json_with_diagnostics(raw_plan2, PlannerOutput)

                if plan_obj is None:
                    # Show diagnostics panel and CONTINUE via fallback (do NOT exit the loop)
                    self.logger.error(f"[PLAN] JSON parse failed.\n{diag}\n--- Retry ---\n{diag2}")
                    console.print(Panel.fit(
                        f"Planner JSON parse failed.\n\n{diag}\n\n--- Retry ---\n{diag2}",
                        title="Parse Error: PLAN", style="red", border_style="red"
                    ))
                    # Fallback to keep loop alive
                    action, controller_reason = self._fallback_policy(target_binary)
                    plan_text = "Planner failed; controller selected next best action."
                    console.print(Panel.fit(f"[PLAN] {plan_text}", style="bold cyan", border_style="cyan"))
                    console.print(Panel.fit(controller_reason, title="Reason", style="dim", border_style="cyan"))
                    self.logger.info(f"[PLAN] {plan_text}")
                    self.logger.info(f"[PLAN][REASON] {controller_reason}")
                    final_answer = None
                else:
                    plan_text = (plan_obj.plan or "").strip()
                    plan_reason = (plan_obj.reason or "").strip()
                    console.print(Panel.fit(f"[PLAN] {plan_text or '(no plan text)'}", style="bold cyan", border_style="cyan"))
                    if plan_reason:
                        console.print(Panel.fit(plan_reason, title="Reason", style="dim", border_style="cyan"))
                    self.logger.info(f"[PLAN] {plan_text}")
                    if plan_reason:
                        self.logger.info(f"[PLAN][REASON] {plan_reason}")
                    final_answer = (plan_obj.final_answer or "").strip() if plan_obj.final_answer else None
                    action = plan_obj.action
            else:
                plan_text = (plan_obj.plan or "").strip()
                plan_reason = (plan_obj.reason or "").strip()
                console.print(Panel.fit(f"[PLAN] {plan_text or '(no plan text)'}", style="bold cyan", border_style="cyan"))
                if plan_reason:
                    console.print(Panel.fit(plan_reason, title="Reason", style="dim", border_style="cyan"))
                self.logger.info(f"[PLAN] {plan_text}")
                if plan_reason:
                    self.logger.info(f"[PLAN][REASON] {plan_reason}")
                final_answer = (plan_obj.final_answer or "").strip() if plan_obj.final_answer else None
                action = plan_obj.action

            # If planner provided final answer directly, finish now
            if isinstance(final_answer, str) and final_answer:
                console.print(Panel(final_answer, title="Final", border_style="green"))
                if plan_obj and (plan_obj.reason or ""):
                    console.print(Panel.fit(plan_obj.reason.strip(), title="Reason", style="dim", border_style="green"))
                self.logger.info(f"[FINAL]\n{final_answer}")
                return final_answer

            # If fallback said "finalize", action will be None; go to finalize path
            if action is None and final_answer is None:
                # Ask model to produce final answer JSON (clean handoff)
                final_prompt = (
                    "Produce the final user-facing answer now, based on all observations so far.\n\n"
                    f"{FINAL_JSON_SPEC}"
                )
                final_messages = [*messages, {"role": "user", "content": final_prompt}]
                raw_final = self._call_llm(final_messages)
                final_obj, fdiag, _ = _parse_json_with_diagnostics(raw_final, FinalOutput)
                if final_obj is None:
                    # Retry once
                    strict_f = [*final_messages, {"role": "user", "content": "STRICT JSON ONLY. Repeat exactly in the specified schema."}]
                    raw_final2 = self._call_llm(strict_f)
                    final_obj, fdiag2, _ = _parse_json_with_diagnostics(raw_final2, FinalOutput)
                    if final_obj is None:
                        self.logger.error(f"[FINAL] JSON parse failed.\n{fdiag}\n--- Retry ---\n{fdiag2}")
                        console.print(Panel.fit(
                            f"Finalizer JSON parse failed.\n\n{fdiag}\n\n--- Retry ---\n{fdiag2}",
                            title="Parse Error: FINAL", style="red", border_style="red"
                        ))
                        return "Finalizer failed."

                final_answer = (final_obj.answer or "").strip()
                final_reason = (final_obj.reason or "").strip()
                console.print(Panel(final_answer or "(no answer) ", title="Final", border_style="green"))
                if final_reason:
                    console.print(Panel.fit(final_reason, title="Reason", style="dim", border_style="green"))
                self.logger.info(f"[FINAL]\n{final_answer}")
                if final_reason:
                    self.logger.info(f"[FINAL][REASON] {final_reason}")
                return final_answer

            # ==========================
            # [ACT]
            # ==========================
            tool = action.tool if PydAvailable else action.get("tool")  # type: ignore
            args = dict(action.args if PydAvailable else action.get("args", {}))  # type: ignore

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
                    tool_content = func(**kwargs)  # JSON string from tool

            # ==========================
            # [OBSERVE]
            # ==========================
            summary = _summarize_tool_result(tool_content)
            obs_line = f"{tool} -> {summary}"
            self.logger.info(f"[OBSERVE] {obs_line}")
            console.print(Panel.fit(f"[OBSERVE] {obs_line}", style="bold green", border_style="green"))

            # Feed a compact observation back to the model
            messages.append({"role": "assistant", "content": f"Observation: {obs_line}"})

            # ==========================
            # [REFLECT]
            # ==========================
            reflect_prompt = (
                "Reflect on the most recent observation and decide whether to continue (another action), "
                "finalize (you can answer now), or replan (change approach).\n\n"
                f"{REFLECT_JSON_SPEC}"
            )
            reflect_messages = [*messages, {"role": "user", "content": reflect_prompt}]
            raw_reflect = self._call_llm(reflect_messages)
            reflect_obj, rdiag, rparts = _parse_json_with_diagnostics(raw_reflect, ReflectOutput)

            if reflect_obj is None:
                # Retry once
                self.logger.warning("[REFLECT] Malformed JSON; retrying with STRICT instruction.")
                strict_r = [*reflect_messages, {"role": "user", "content": "STRICT JSON ONLY. Repeat exactly in the specified JSON schema."}]
                raw_reflect2 = self._call_llm(strict_r)
                reflect_obj, rdiag2, _ = _parse_json_with_diagnostics(raw_reflect2, ReflectOutput)
                if reflect_obj is None:
                    self.logger.error(f"[REFLECT] JSON parse failed.\n{rdiag}\n--- Retry ---\n{rdiag2}")
                    console.print(Panel.fit(
                        f"Reflector JSON parse failed.\n\n{rdiag}\n\n--- Retry ---\n{rdiag2}",
                        title="Parse Error: REFLECT", style="red", border_style="red"
                    ))
                    # If reflector fails, keep iterating using controller guidance
                    reflect_obj = ReflectOutput() if PydAvailable else {"reflection": "", "decision": "continue", "reason": ""}  # type: ignore
                    if not PydAvailable:
                        reflect_obj["decision"] = "continue"  # type: ignore

            reflect_text = (reflect_obj.reflection or "").strip() if PydAvailable else (reflect_obj.get("reflection","").strip())  # type: ignore
            reflect_reason = (reflect_obj.reason or "").strip() if PydAvailable else (reflect_obj.get("reason","").strip())  # type: ignore
            console.print(Panel.fit(f"[REFLECT] {reflect_text or '(no reflection)'}", style="bold magenta", border_style="magenta"))
            if reflect_reason:
                console.print(Panel.fit(reflect_reason, title="Reason", style="dim", border_style="magenta"))
            self.logger.info(f"[REFLECT] {reflect_text}")
            if reflect_reason:
                self.logger.info(f"[REFLECT][REASON] {reflect_reason}")

            # Decision
            decision = (reflect_obj.decision or "").strip().lower() if PydAvailable else (reflect_obj.get("decision","").strip().lower())  # type: ignore
            if decision == "finalize":
                final_prompt = (
                    "Produce the final user-facing answer now, based on all observations so far.\n\n"
                    f"{FINAL_JSON_SPEC}"
                )
                final_messages = [*messages, {"role": "user", "content": final_prompt}]
                raw_final = self._call_llm(final_messages)
                final_obj, fdiag, fparts = _parse_json_with_diagnostics(raw_final, FinalOutput)
                if final_obj is None:
                    # Retry once
                    strict_f = [*final_messages, {"role": "user", "content": "STRICT JSON ONLY. Repeat exactly in the specified schema."}]
                    raw_final2 = self._call_llm(strict_f)
                    final_obj, fdiag2, _ = _parse_json_with_diagnostics(raw_final2, FinalOutput)
                    if final_obj is None:
                        self.logger.error(f"[FINAL] JSON parse failed.\n{fdiag}\n--- Retry ---\n{fdiag2}")
                        console.print(Panel.fit(
                            f"Finalizer JSON parse failed.\n\n{fdiag}\n\n--- Retry ---\n{fdiag2}",
                            title="Parse Error: FINAL", style="red", border_style="red"
                        ))
                        return "Finalizer failed."

                final_answer = (final_obj.answer or "").strip() if PydAvailable else (final_obj.get("answer","").strip())  # type: ignore
                final_reason = (final_obj.reason or "").strip() if PydAvailable else (final_obj.get("reason","").strip())  # type: ignore
                console.print(Panel(final_answer or "(no answer) ", title="Final", border_style="green"))
                if final_reason:
                    console.print(Panel.fit(final_reason, title="Reason", style="dim", border_style="green"))
                self.logger.info(f"[FINAL]\n{final_answer}")
                if final_reason:
                    self.logger.info(f"[FINAL][REASON] {final_reason}")
                return final_answer

            # continue or replan → iterate

        self.logger.warning("[AGENT] Max iterations reached without a final answer.")
        return "Stopped after max iterations without a final answer."

# --------------------------------------------------------------------------------------
# Typer commands
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

        # Fallthrough: send plain question (iterative controller will still run)
        user_msg = cmd
        final = sess.ask_once(user_msg)
        console.print(Panel(final or "(no answer)", title="Response", border_style="green"))

def main():
    app()

if __name__ == "__main__":
    sys.exit(main())
