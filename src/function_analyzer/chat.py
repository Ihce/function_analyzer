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
from rich.syntax import Syntax
from rich.text import Text

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

You MUST use these tools to gather information about binaries:
- binary_loader(file_path: string) — loads the binary and exposes .text, architecture, and base
- boundary_detector(model_path?) — predicts function boundaries
- disassembler() — disassembles known functions; takes NO arguments (pass {})

Your process:
1. ALWAYS start by using tools to gather concrete data about the binary
2. Continue using tools until you have exhausted their capabilities or gathered sufficient information
3. Once tools have provided enough data, analyze and reason about what you've discovered
4. Provide your final answer based on the evidence collected

You cannot answer questions about binaries without first using tools to examine them.
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
# Improved JSON instructions with examples
# --------------------------------------------------------------------------------------
PLANNER_JSON_SPEC = """Return ONLY valid JSON matching this exact structure:
{
  "plan": "one short sentence describing the next step",
  "action": {"tool": "binary_loader|boundary_detector|disassembler", "args": {}} or null,
  "final_answer": null or "string if ready to answer",
  "reason": "why this plan/action makes sense",
  "confidence": 0.0 to 1.0 (e.g. 0.3, 0.7, 0.95)
}

Rules:
- Use tools to gather data first, THEN reason about findings
- If you need more info, set action with a tool and final_answer=null
- If tools have provided enough data and you're ready to analyze/answer, set action=null and final_answer="your analysis"
- For disassembler, args MUST be exactly {}
- confidence should vary based on certainty (0.3=low, 0.7=moderate, 0.95=high)
- Start with { and end with }. No markdown, no extra text.

Example responses:
{"plan": "Load the binary to access its sections", "action": {"tool": "binary_loader", "args": {"file_path": "/path/to/binary"}}, "final_answer": null, "reason": "Need binary data to analyze", "confidence": 0.9}
{"plan": "Analyze gathered data for buffer overflow patterns", "action": null, "final_answer": "Based on disassembly analysis, I found 3 potential buffer overflow vulnerabilities...", "reason": "Have sufficient data from tools to provide analysis", "confidence": 0.85}"""

REFLECT_JSON_SPEC = """Return ONLY valid JSON:
{
  "reflection": "what the observation tells us",
  "decision": "continue" (more tools needed) or "finalize" (enough data, ready to analyze) or "replan" (change approach),
  "reason": "why this decision",
  "confidence": 0.0 to 1.0 (vary based on certainty)
}

Decision guidelines:
- "continue": Need more tool data before analysis (e.g., haven't loaded binary yet, haven't detected functions, haven't disassembled)
- "finalize": Tools have provided sufficient data - switch to reasoning/analysis mode
  * ALWAYS finalize after disassembler has been called (you have the assembly code)
  * Finalize when you have all the raw data needed to answer the question
- "replan": Current approach isn't working, try different tools/strategy

IMPORTANT: Once you have disassembly output, you have ALL the data tools can provide. Switch to "finalize" to analyze the code.

Example:
{"reflection": "Disassembly shows 238 functions with assembly code visible", "decision": "finalize", "reason": "Have all necessary data to analyze for vulnerabilities", "confidence": 0.9}"""

FINAL_JSON_SPEC = """Return ONLY valid JSON:
{
  "answer": "your detailed analysis based on the tool data gathered",
  "reason": "summary of evidence from tools that supports your analysis",
  "confidence": 0.0 to 1.0 (vary based on certainty)
}

Now that tools have provided data, use your reasoning capabilities to:
- Analyze patterns in the disassembly
- Identify vulnerabilities or issues
- Explain what the binary does
- Answer the user's specific question

Example:
{"answer": "Analysis of the 238 functions reveals 3 potential buffer overflow vulnerabilities: Function at 0x401080 uses strcpy without bounds checking...", "reason": "Based on disassembly showing unsafe string operations and stack manipulation patterns", "confidence": 0.85}"""

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

    # Always add file handler if log_file is provided
    if log_file:
        fh = logging.FileHandler(log_file, mode="a")
        fh.setFormatter(fmt_file)
        logger.addHandler(fh)

    # Only add console handler if explicitly requested (which we won't do anymore)
    if to_console:
        fmt_console = logging.Formatter('[%(levelname)s] %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(fmt_console)
        # Only show warnings and errors on console, not info/debug
        ch.setLevel(logging.WARNING)
        logger.addHandler(ch)

    return logger

def _extract_target_binary(text: str) -> Optional[str]:
    for line in text.splitlines():
        if line.lower().startswith("target binary:"):
            val = line.split(":", 1)[1].strip()
            if val:
                return val
    return None

# -------- Improved JSON parsing with multiple strategies --------
def _strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    # Remove markdown code fences
    s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE | re.MULTILINE)
    s = re.sub(r'\s*```$', '', s, flags=re.MULTILINE)
    return s.strip()

def _extract_json_block(text: str) -> Optional[str]:
    """Extract JSON block with multiple strategies."""
    if not text:
        return None

    # Clean up the text first
    text = _strip_code_fences(text)

    # Strategy 1: If it's already valid JSON
    if text.startswith("{") and text.endswith("}"):
        return text

    # Strategy 2: Find JSON object boundaries
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

    # Strategy 3: If we found a start but no proper end, try to extract what we can
    if start >= 0:
        # Look for the last closing brace
        end = text.rfind("}")
        if end > start:
            return text[start:end+1].strip()

    return None

def _repair_json(text: str) -> Optional[str]:
    """Try to repair common JSON errors."""
    if not text:
        return None

    # Remove trailing commas
    text = re.sub(r',\s*}', '}', text)
    text = re.sub(r',\s*]', ']', text)

    # Fix single quotes to double quotes
    # This is risky but sometimes necessary
    text = re.sub(r"'([^']*)'", r'"\1"', text)

    # Ensure boolean values are lowercase
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)

    return text

def _parse_json_with_fallbacks(raw: str, expected_fields: List[str] = None) -> Optional[Dict[str, Any]]:
    """Parse JSON with multiple fallback strategies."""
    if not raw:
        return None

    # Try to extract JSON block
    block = _extract_json_block(raw)
    if not block:
        return None

    # Try parsing as-is
    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    # Try repairing common issues
    repaired = _repair_json(block)
    if repaired:
        try:
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Last resort: try to extract key-value pairs manually
    if expected_fields:
        result = {}
        for field in expected_fields:
            # Look for patterns like "field": "value" or "field": value
            pattern = rf'"{field}"\s*:\s*("(?:[^"\\]|\\.)*"|null|true|false|\d+\.?\d*|\[.*?\]|\{{.*?\}})'
            match = re.search(pattern, block, re.DOTALL)
            if match:
                try:
                    result[field] = json.loads(match.group(1))
                except:
                    result[field] = match.group(1).strip('"')

        if result:
            return result

    return None

# --------------------------------------------------------------------------------------
# Session with improved error handling and reasoning display
# --------------------------------------------------------------------------------------
class Session:
    def __init__(
        self,
        binary: Optional[str],
        boundary_ckpt: Optional[str],
        model_id: str,
        vllm_url: str,  # Renamed parameter
        verbose: bool,
        log_file: Optional[str],
        no_console: bool,
        show_reasoning: bool = True,
    ) -> None:
        self.logger = _setup_logger("chat", log_file, verbose, to_console=False)
        self.agent = BinaryAnalysisAgent(
            vllm_url=vllm_url,  # Updated parameter name
            model_id=model_id,
            boundary_ckpt=boundary_ckpt or "models/function_boundary.pth",
            verbose=verbose,
            log_file=log_file,
            log_to_console=False,
        )
        self.system_prompt = SYSTEM_PROMPT
        self.binary_hint = binary
        self.verbose = verbose
        self.show_reasoning = show_reasoning
        self.retry_count = 3  # Number of retries for LLM calls

        self.logger.info("=" * 80)
        self.logger.info(f"CHAT SESSION STARTED - {datetime.now()}")
        self.logger.info(f"Model: {model_id}")
        self.logger.info(f"vLLM URL: {vllm_url}")
        self.logger.info(f"Binary hint: {binary or 'None'}")
        self.logger.info("=" * 80)

    def _call_llm(self, messages: list[dict[str, str]], json_mode: bool = False) -> str:
        """LLM call optimized for vLLM."""
        try:
            params = {
                "model": self.agent.model_id,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 2048,
                "timeout": 60,  # vLLM is typically faster
            }

            if json_mode:
                # Try JSON mode, fall back if not supported
                try:
                    params["response_format"] = {"type": "json_object"}
                    resp = self.agent.client.chat.completions.create(**params)
                except Exception as e:
                    if "json" in str(e).lower() or "format" in str(e).lower():
                        self.logger.warning("JSON mode not supported, using standard mode")
                        params.pop("response_format", None)
                        resp = self.agent.client.chat.completions.create(**params)
                    else:
                        raise
            else:
                resp = self.agent.client.chat.completions.create(**params)

            content = (resp.choices[0].message.content or "").strip()

            if self.verbose and content:
                self.logger.debug(f"[LLM Response] Length: {len(content)} chars")
                self.logger.debug(f"[LLM Response] First 500: {content[:500]}")

            return content

        except Exception as e:
            self.logger.error(f"[LLM Error]: {e}")
            return ""

    def _display_reasoning(self, stage: str, content: Dict[str, Any], iteration: int) -> str:
        """Create formatted reasoning display."""
        parts = []

        if stage == "PLAN":
            parts.append(f"[bold cyan]PLAN:[/bold cyan] {content.get('plan', 'N/A')}")
            if self.show_reasoning and content.get('reason'):
                parts.append(f"[dim]→ Reasoning: {content['reason']}[/dim]")
            if content.get('confidence') is not None:
                conf = content['confidence']
                # Color code confidence
                if conf >= 0.8:
                    conf_color = "green"
                elif conf >= 0.5:
                    conf_color = "yellow"
                else:
                    conf_color = "red"
                parts.append(f"[dim]→ Confidence: [{conf_color}]{conf:.1%}[/{conf_color}][/dim]")

        elif stage == "REFLECT":
            parts.append(f"[bold magenta]REFLECT:[/bold magenta] {content.get('reflection', 'N/A')}")
            if self.show_reasoning and content.get('reason'):
                parts.append(f"[dim]→ Reasoning: {content['reason']}[/dim]")
            parts.append(f"[dim]→ Decision: {content.get('decision', 'continue')}[/dim]")
            if content.get('confidence') is not None:
                conf = content['confidence']
                if conf >= 0.8:
                    conf_color = "green"
                elif conf >= 0.5:
                    conf_color = "yellow"
                else:
                    conf_color = "red"
                parts.append(f"[dim]→ Confidence: [{conf_color}]{conf:.1%}[/{conf_color}][/dim]")

        return "\n".join(parts)

    def ask_once(self, user_msg: str, max_iterations: int = 10) -> str:
        """Improved PLAN → ACT → OBSERVE → REFLECT loop with better error handling."""
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_msg},
        ]
        target_binary = _extract_target_binary(user_msg) or self.binary_hint

        # Track tool usage to detect redundancy
        tool_history = []
        last_tool_results = {}

        for iteration in range(1, max_iterations + 1):
            self.logger.info(f"[ITER {iteration}] " + "=" * 50)

            # Create a group for this iteration
            iteration_group = []

            # Add context about what tools have been used
            tools_used_summary = ""
            if tool_history:
                unique_tools = list(set(tool_history))
                tools_used_summary = f"\n\nTools already used: {', '.join(unique_tools)}"
                if len(tool_history) > len(unique_tools):
                    tools_used_summary += f"\n(Some tools have been called multiple times - avoid redundant calls)"

            # [PLAN] with JSON mode
            planner_prompt = (
                "Decide the very next step toward the objective.\n"
                "Choose ANY available tool in ANY order, or provide a final answer if ready.\n"
                f"{tools_used_summary}\n\n"
                f"{PLANNER_JSON_SPEC}"
            )
            plan_messages = [*messages, {"role": "user", "content": planner_prompt}]

            # Use JSON mode for guaranteed valid JSON
            raw_plan = self._call_llm(plan_messages, json_mode=True)
            plan_obj = None

            if raw_plan:
                try:
                    plan_obj = json.loads(raw_plan)
                except json.JSONDecodeError as e:
                    self.logger.error(f"[PLAN] JSON decode error despite JSON mode: {e}")
                    self.logger.error(f"[PLAN] Raw response: {repr(raw_plan[:500])}")

            if not plan_obj:
                # If JSON mode isn't working, try without it
                self.logger.warning("[PLAN] Retrying without JSON mode")
                raw_plan = self._call_llm(plan_messages, json_mode=False)
                plan_obj = _parse_json_with_fallbacks(
                    raw_plan,
                    expected_fields=["plan", "action", "final_answer", "reason", "confidence"]
                )

            if not plan_obj:
                self.logger.error(f"[PLAN] Failed to parse response")
                error_msg = "[red]Failed to parse plan. Check if your vLLM version supports JSON mode.[/red]"
                console.print(Panel(error_msg, title=f"Iteration {iteration} - Error", border_style="red"))
                return "Failed to parse plan. Consider updating vLLM or using a different model."

            # Display planning with reasoning
            plan_display = self._display_reasoning("PLAN", plan_obj, iteration)
            iteration_group.append(plan_display)

            self.logger.info(f"[PLAN] {plan_obj.get('plan', 'N/A')}")
            if plan_obj.get('reason'):
                self.logger.info(f"[PLAN][REASON] {plan_obj['reason']}")

            # Check for final answer
            final_answer = plan_obj.get("final_answer")
            if final_answer:
                iteration_group.append(f"[bold green]FINAL:[/bold green] {final_answer}")
                console.print(Panel("\n\n".join(iteration_group),
                                  title=f"Iteration {iteration} - Complete",
                                  border_style="green"))
                self.logger.info(f"[FINAL]\n{final_answer}")
                return final_answer

            # Get action
            action = plan_obj.get("action")
            if not action:
                # Try to get final answer with JSON mode
                final_prompt = (
                    "Provide your final answer based on all observations.\n\n"
                    f"{FINAL_JSON_SPEC}"
                )
                final_messages = [*messages, {"role": "user", "content": final_prompt}]

                # Use JSON mode for guaranteed valid JSON
                raw_final = self._call_llm(final_messages, json_mode=True)
                final_obj = None

                if raw_final:
                    try:
                        final_obj = json.loads(raw_final)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"[FINAL] JSON decode error despite JSON mode: {e}")

                if not final_obj:
                    # Try without JSON mode as fallback
                    raw_final = self._call_llm(final_messages, json_mode=False)
                    final_obj = _parse_json_with_fallbacks(
                        raw_final,
                        expected_fields=["answer", "reason", "confidence"]
                    )

                if final_obj:
                    answer = final_obj.get("answer", "")
                    if self.show_reasoning and final_obj.get("reason"):
                        iteration_group.append(f"[bold green]FINAL:[/bold green] {answer}")
                        iteration_group.append(f"[dim]→ Reasoning: {final_obj['reason']}[/dim]")
                    else:
                        iteration_group.append(f"[bold green]FINAL:[/bold green] {answer}")

                    console.print(Panel("\n\n".join(iteration_group),
                                      title=f"Iteration {iteration} - Complete",
                                      border_style="green"))
                    self.logger.info(f"[FINAL]\n{answer}")
                    return answer
                else:
                    self.logger.error(f"[FINAL] Failed to parse")
                    iteration_group.append(f"[red]Failed to parse final answer[/red]")
                    console.print(Panel("\n\n".join(iteration_group),
                                      title=f"Iteration {iteration} - Error",
                                      border_style="red"))
                    return "Failed to get final answer."

            # [ACT] - Execute the action
            tool = action.get("tool")
            args = action.get("args", {})

            # Check if this is a redundant call
            tool_key = f"{tool}_{json.dumps(args, sort_keys=True)}"
            if tool_key in last_tool_results and len(tool_history) >= 3:
                # We've called this exact tool with these args before
                self.logger.warning(f"[ACT] Redundant tool call detected: {tool}")
                tool_content = last_tool_results[tool_key]
                iteration_group.append(f"[yellow]⚠ Redundant tool call - using cached result[/yellow]")
            else:
                # Autofill binary path for loader if missing
                if tool == "binary_loader" and "file_path" not in args and target_binary:
                    args["file_path"] = target_binary

                args = _normalize_args(tool, args)

                act_content = f"[bold yellow]ACT:[/bold yellow] {tool}({', '.join(f'{k}={v}' for k, v in args.items())})"
                iteration_group.append(act_content)

                self.logger.info(f"[ACT] {tool} args={args}")

                func_tuple = self.agent._tools.get(tool)
                if not func_tuple:
                    tool_content = json.dumps({"error": f"Unknown tool: {tool}"})
                else:
                    func, ArgsModel = func_tuple
                    try:
                        validated_args = ArgsModel(**args)
                        if hasattr(validated_args, 'dict'):
                            kwargs = validated_args.dict()
                        else:
                            kwargs = validated_args.model_dump()
                        self.logger.info(f"[TOOL EXEC] {tool}")
                        tool_content = func(**kwargs)
                        # Cache the result
                        last_tool_results[tool_key] = tool_content
                    except Exception as e:
                        tool_content = json.dumps({"error": f"Tool error: {e}"})

                # Track tool usage
                tool_history.append(tool)

            # [OBSERVE] — show a human summary in the console,
            #            but ADD THE FULL RAW TOOL OUTPUT to the model context.
            summary = _summarize_tool_result(tool_content)
            obs_line = f"{tool} → {summary}"
            observe_content = f"[bold green]OBSERVE:[/bold green] {obs_line}"
            iteration_group.append(observe_content)
            self.logger.info(f"[OBSERVE] {obs_line}")

            # Feed the FULL tool output into the chat so the model can actually use it.
            # Keep it as plain text; models handle raw JSON fine when embedded in text.
            if tool_content:
                # If extremely large, you could chunk here; for now, send as one block.
                messages.append({
                    "role": "assistant",
                    "content": f"Observation [{tool}] — FULL OUTPUT FOLLOWS:\n{tool_content}"
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": f"Observation [{tool}] — (no output)"
                })

            # [REFLECT] with JSON mode
            reflect_context = ""
            if tool_history.count(tool) > 2:
                reflect_context = f"\nNOTE: {tool} has been called {tool_history.count(tool)} times. If you have sufficient data, finalize instead of repeating.\n"

            reflect_prompt = (
                f"Reflect on the observation and decide next step.{reflect_context}\n\n"
                f"{REFLECT_JSON_SPEC}"
            )
            reflect_messages = [*messages, {"role": "user", "content": reflect_prompt}]

            # Use JSON mode for guaranteed valid JSON
            raw_reflect = self._call_llm(reflect_messages, json_mode=True)
            reflect_obj = None

            if raw_reflect:
                try:
                    reflect_obj = json.loads(raw_reflect)
                except json.JSONDecodeError as e:
                    self.logger.error(f"[REFLECT] JSON decode error despite JSON mode: {e}")
                    self.logger.error(f"[REFLECT] Raw response: {repr(raw_reflect[:500])}")

            if not reflect_obj:
                # Try without JSON mode as fallback
                self.logger.warning("[REFLECT] Retrying without JSON mode")
                raw_reflect = self._call_llm(reflect_messages, json_mode=False)
                reflect_obj = _parse_json_with_fallbacks(
                    raw_reflect,
                    expected_fields=["reflection", "decision", "reason", "confidence"]
                )

            if not reflect_obj:
                self.logger.error(f"[REFLECT] Failed to parse - using default")
                # Default to continue
                reflect_obj = {"reflection": "Processing observation", "decision": "continue"}

            # Display reflection with reasoning
            reflect_display = self._display_reasoning("REFLECT", reflect_obj, iteration)
            iteration_group.append(reflect_display)

            self.logger.info(f"[REFLECT] {reflect_obj.get('reflection', 'N/A')}")
            if reflect_obj.get('reason'):
                self.logger.info(f"[REFLECT][REASON] {reflect_obj['reason']}")

            decision = reflect_obj.get("decision", "continue").lower()

            # Force finalize if we've called disassembler 3+ times
            if tool == "disassembler" and tool_history.count("disassembler") >= 3:
                self.logger.warning("[REFLECT] Forcing finalize - disassembler called 3+ times")
                decision = "finalize"
                iteration_group.append(f"[yellow]⚠ Tool exhaustion detected - switching to analysis mode[/yellow]")

            # Display the iteration
            border_color = "cyan" if decision == "continue" else "yellow"
            console.print(Panel("\n\n".join(iteration_group),
                              title=f"Iteration {iteration}",
                              border_style=border_color))

            # If decision is finalize, get final answer with JSON mode
            if decision == "finalize":
                final_prompt = (
                    "Provide your final answer based on all observations.\n\n"
                    f"{FINAL_JSON_SPEC}"
                )
                final_messages = [*messages, {"role": "user", "content": final_prompt}]

                # Use JSON mode for guaranteed valid JSON
                raw_final = self._call_llm(final_messages, json_mode=True)
                final_obj = None

                if raw_final:
                    try:
                        final_obj = json.loads(raw_final)
                    except json.JSONDecodeError as e:
                        self.logger.error(f"[FINAL] JSON decode error despite JSON mode: {e}")
                        self.logger.error(f"[FINAL] Raw response: {repr(raw_final[:500])}")

                if not final_obj:
                    # Try without JSON mode as fallback
                    self.logger.warning("[FINAL] Retrying without JSON mode")
                    raw_final = self._call_llm(final_messages, json_mode=False)
                    final_obj = _parse_json_with_fallbacks(
                        raw_final,
                        expected_fields=["answer", "reason", "confidence"]
                    )

                if final_obj:
                    answer = final_obj.get("answer", "")
                    final_panel_content = f"[bold green]FINAL ANSWER:[/bold green]\n\n{answer}"

                    if self.show_reasoning and final_obj.get("reason"):
                        final_panel_content += f"\n\n[dim]Reasoning: {final_obj['reason']}[/dim]"
                    if final_obj.get("confidence") is not None:
                        final_panel_content += f"\n[dim]Confidence: {final_obj['confidence']:.1%}[/dim]"

                    console.print(Panel(final_panel_content,
                                      title="Analysis Complete",
                                      border_style="green",
                                      box=box.DOUBLE))
                    self.logger.info(f"[FINAL]\n{answer}")
                    return answer
                else:
                    self.logger.error(f"[FINAL] Failed to parse")
                    console.print(Panel(f"[red]Failed to parse final answer[/red]",
                                      title="Error",
                                      border_style="red"))
                    return "Failed to get final answer."

        self.logger.warning("[AGENT] Max iterations reached.")
        console.print(Panel("[yellow]Max iterations reached without final answer.[/yellow]",
                          title="Incomplete",
                          border_style="yellow"))
        return "Max iterations reached without final answer."

# --------------------------------------------------------------------------------------
# Updated Typer commands with reasoning flag
# --------------------------------------------------------------------------------------
def _common_session(
    binary: Optional[str],
    boundary_ckpt: Optional[str],
    model_id: str,
    vllm_url: str,
    verbose: bool,
    log_dir: str,
    log_file: Optional[str],
    no_console: bool,
    show_reasoning: bool,
) -> Session:
    log_path = log_file or _default_log_path(log_dir)
    return Session(binary, boundary_ckpt, model_id, vllm_url, verbose, log_path, no_console, show_reasoning)

@app.command(help="Analyze a binary and answer a question (iterative).")
def ask(
    binary: str = typer.Argument(..., help="Path to the binary file"),
    question: str = typer.Argument(..., help="Question to answer about the binary"),
    model_id: str = typer.Option(None, "--model-id", help="Model ID (defaults to env or gpt-oss)"),
    vllm_url: str = typer.Option(None, "--vllm-url", help="vLLM OpenAI-compatible URL"),
    ollama_url: str = typer.Option(None, "--ollama-url", help="(Deprecated) Use --vllm-url"),
    boundary_ckpt: Optional[str] = typer.Option(None, "--boundary-ckpt", help="Path to boundary model"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Verbose logging"),
    log_dir: str = typer.Option("./logs", "--log-dir", help="Log directory"),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Custom log file path"),
    show_reasoning: bool = typer.Option(True, "--show-reasoning/--hide-reasoning", help="Show model reasoning"),
):
    # Handle backwards compatibility
    url = vllm_url or ollama_url or os.getenv("VLLM_URL", "http://localhost:8000/v1")
    model = model_id or os.getenv("MODEL_ID", "openai/gpt-oss-20b")

    sess = _common_session(binary, boundary_ckpt, model, url, verbose, log_dir, log_file, True, show_reasoning)
    user_msg = f"""Target binary: {binary}

Objective: {question}

Iterate with PLAN → ACT → OBSERVE → REFLECT until you can answer confidently. Output your final answer only when asked to finalize."""
    console.print(Panel.fit(f"[b]{os.path.basename(binary)}[/b] • {question}", title="ASK", style="magenta"))
    final = sess.ask_once(user_msg)
    console.print(Panel(final or "(no answer)", title="Answer", border_style="green"))

@app.command(help="Open a binary and summarize what it does (iterative).")
def open(
    binary: str = typer.Argument(..., help="Path to the binary file"),
    model_id: str = typer.Option(None, "--model-id", help="Model ID (defaults to env or gpt-oss)"),
    vllm_url: str = typer.Option(None, "--vllm-url", help="vLLM OpenAI-compatible URL"),
    ollama_url: str = typer.Option(None, "--ollama-url", help="(Deprecated) Use --vllm-url"),
    boundary_ckpt: Optional[str] = typer.Option(None, "--boundary-ckpt"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
    log_dir: str = typer.Option("./logs", "--log-dir"),
    log_file: Optional[str] = typer.Option(None, "--log-file"),
    show_reasoning: bool = typer.Option(True, "--show-reasoning/--hide-reasoning"),
):
    # Handle backwards compatibility
    url = vllm_url or ollama_url or os.getenv("VLLM_URL", "http://localhost:8000/v1")
    model = model_id or os.getenv("MODEL_ID", "openai/gpt-oss-20b")

    sess = _common_session(binary, boundary_ckpt, model, url, verbose, log_dir, log_file, True, show_reasoning)
    user_msg = f"""Target binary: {binary}

Objective: Summarize what this program does.

Iterate with PLAN → ACT → OBSERVE → REFLECT until a concise summary can be finalized. Output your final answer only when asked to finalize."""
    console.print(Panel.fit(f"[b]{os.path.basename(binary)}[/b]", title="OPEN", style="magenta"))
    final = sess.ask_once(user_msg)
    console.print(Panel(final or "(no answer)", title="Summary", border_style="green"))

@app.command(help="List available tools.")
def tools(
    model_id: str = typer.Option(None, "--model-id"),
    vllm_url: str = typer.Option(None, "--vllm-url"),
    ollama_url: str = typer.Option(None, "--ollama-url"),
    boundary_ckpt: Optional[str] = typer.Option(None, "--boundary-ckpt"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
    log_dir: str = typer.Option("./logs", "--log-dir"),
    log_file: Optional[str] = typer.Option(None, "--log-file"),
):
    url = vllm_url or ollama_url or os.getenv("VLLM_URL", "http://localhost:8000/v1")
    model = model_id or os.getenv("MODEL_ID", "openai/gpt-oss-20b")

    sess = _common_session(None, boundary_ckpt, model, url, verbose, log_dir, log_file, True, True)
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
    model_id: str = typer.Option(None, "--model-id"),
    vllm_url: str = typer.Option(None, "--vllm-url"),
    ollama_url: str = typer.Option(None, "--ollama-url"),
    boundary_ckpt: Optional[str] = typer.Option(None, "--boundary-ckpt"),
    binary: Optional[str] = typer.Option(None, "--binary", help="Default binary hint"),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
    log_dir: str = typer.Option("./logs", "--log-dir"),
    log_file: Optional[str] = typer.Option(None, "--log-file"),
    show_reasoning: bool = typer.Option(True, "--show-reasoning/--hide-reasoning"),
):
    url = vllm_url or ollama_url or os.getenv("VLLM_URL", "http://localhost:8000/v1")
    model = model_id or os.getenv("MODEL_ID", "openai/gpt-oss-20b")

    sess = _common_session(binary, boundary_ckpt, model, url, verbose, log_dir, log_file, True, show_reasoning)
    console.print(Panel.fit("Commands: ask, open, tools, reset, verbose, reasoning, exit", title="REPL", style="magenta"))

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
            console.print("Commands: ask <binary> <question> | open <binary> | tools | reset | verbose | reasoning | exit")
            continue
        if cmd == "verbose":
            # Toggle verbose mode for both session and agent
            sess.verbose = not sess.verbose
            sess.logger.setLevel(logging.DEBUG if sess.verbose else logging.INFO)
            sess.agent.log.setLevel(logging.DEBUG if sess.verbose else logging.INFO)

            status = 'ON' if sess.verbose else 'OFF'
            console.print(f"[yellow]Verbose: {status}[/yellow]")

            if sess.verbose:
                console.print("[dim]Raw LLM responses will now be logged to file[/dim]")
            else:
                console.print("[dim]Raw LLM response logging disabled[/dim]")

            continue
        if cmd == "reasoning":
            sess.show_reasoning = not sess.show_reasoning
            console.print(f"Reasoning display: {'ON' if sess.show_reasoning else 'OFF'}")
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

        # Fallthrough: send plain question (could be non-binary related)
        user_msg = cmd
        console.print(Panel.fit(f"[dim]Processing: {cmd[:80]}{'...' if len(cmd) > 80 else ''}[/dim]",
                               title="Query", style="cyan"))
        final = sess.ask_once(user_msg)
        console.print(Panel(final or "(no answer)", title="Response", border_style="green"))

def main():
    app()

if __name__ == "__main__":
    sys.exit(main())
