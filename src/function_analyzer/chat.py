#!/usr/bin/env python3
import argparse
import json
import sys
import os
from datetime import datetime
import logging
from .agent import BinaryAnalysisAgent


# Enhanced system prompt that explicitly guides tool usage
SYSTEM_PROMPT = """You are a reverse-engineering assistant specialized in binary analysis.

You have access to THREE tools that you MUST use in sequence to analyze binary files:

1. **binary_loader** - ALWAYS use this FIRST to load PE binary files
   - Takes a file_path parameter
   - Extracts the .text section containing executable code
   - Returns binary data, architecture, and base address

2. **boundary_detector** - Use this SECOND after binary_loader succeeds
   - Uses neural networks to detect function boundaries
   - Returns a list of functions with their addresses
   - Optional model_path parameter (usually not needed)

3. **disassembler** - Use this THIRD after boundary_detector succeeds
   - Disassembles the detected functions into assembly code
   - Returns human-readable assembly instructions
   - No parameters needed (uses detected functions)

IMPORTANT WORKFLOW:
- When given a binary file path and a question about it, ALWAYS:
  1. First call binary_loader with the file path
  2. Then call boundary_detector to find functions
  3. Then call disassembler to get assembly code
  4. Finally, analyze the results to answer the user's question

You are an expert at x86/x64 assembly and can:
- Add detailed comments explaining what assembly instructions do
- Identify common patterns and programming constructs
- Explain function purposes based on their assembly code
- Format and present assembly code clearly

Always use the tools when asked about a binary file. Never refuse to analyze a binary."""


def setup_log_file(args) -> str:
    """Set up the log file path for chat session."""
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.log_file:
        log_file = args.log_file
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(args.log_dir, f"chat_session_{timestamp}.log")

    return log_file


def run_chat(binary: str | None, boundary_ckpt: str | None, model_id: str, ollama_url: str,
             verbose: bool, log_file: str, no_console: bool, enhanced_prompts: bool):

    agent = BinaryAnalysisAgent(
        ollama_url=ollama_url,
        model_id=model_id,
        boundary_ckpt=boundary_ckpt,
        verbose=verbose,
        log_file=log_file,
        log_to_console=not no_console,
    )

    # Always use the enhanced system prompt
    system_prompt = SYSTEM_PROMPT

    # Initialize conversation with system prompt
    messages = [{"role": "system", "content": system_prompt}]

    # Setup logging
    logger = logging.getLogger("chat")
    logger.setLevel(logging.INFO if not verbose else logging.DEBUG)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        logger.addHandler(file_handler)

    # Console handler
    if not no_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
        logger.addHandler(console_handler)

    logger.info("="*80)
    logger.info(f"CHAT SESSION STARTED - {datetime.now()}")
    logger.info(f"Model: {model_id}")
    logger.info(f"Binary hint: {binary if binary else 'None provided'}")
    logger.info(f"Enhanced prompts: {enhanced_prompts}")
    logger.info("="*80)

    print(f"Chat started. Log file: {log_file}")
    print("\nCommands:")
    print("  /help          — show this help")
    print("  /ask <binary_path> <question>  — analyze a binary and answer a question")
    print("  /open <path>   — analyze a binary file (basic analysis)")
    print("  /tools         — list available tools")
    print("  /state         — show current analysis state")
    print("  /reset         — clear conversation and state")
    print("  /verbose       — toggle verbose mode")
    print("  /log           — toggle file logging on/off")
    print("  /exit          — quit\n")

    if binary:
        print(f"Hint: You have --binary {binary}. Try:")
        print(f"  '/ask {binary} What does this binary do?'")
        print(f"  '/ask {binary} Show me the first 10 functions with comments'\n")

    file_logging_enabled = True

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            logger.info("Chat session ended by user interrupt")
            break

        if not user:
            continue

        logger.info(f"[USER INPUT] {user}")

        # Handle commands
        if user == "/exit":
            logger.info("Chat session ended by /exit command")
            break

        elif user == "/help":
            print("\nCommands:")
            print("  /ask <binary_path> <question>  — analyze a binary and answer a question")
            print("     Example: /ask /path/to/binary.bin Show the first 5 functions with comments")
            print("  /open <path>   — analyze a binary file (basic analysis)")
            print("  /tools         — list available tools")
            print("  /state         — show current analysis state")
            print("  /reset         — clear conversation and state")
            print("  /verbose       — toggle verbose mode")
            print("  /log           — toggle file logging on/off")
            print("  /exit          — quit\n")
            continue

        elif user == "/tools":
            print("\nAvailable tools:")
            for i, tool in enumerate(agent.functions, 1):
                print(f"{i}. {tool['name']}: {tool.get('description', 'No description')[:100]}...")
            print()
            continue

        elif user == "/state":
            print("\nCurrent state:")
            print(f"  Binary loaded: {'Yes' if agent.state.get('binary') else 'No'}")
            if agent.state.get('binary'):
                print(f"    Architecture: {agent.state['binary'].get('architecture')}")
                print(f"    Base address: 0x{agent.state['binary'].get('base_address', 0):08X}")
            print(f"  Functions detected: {'Yes' if agent.state.get('functions') else 'No'}")
            if agent.state.get('functions'):
                print(f"    Count: {len(agent.state['functions'])}")
            print(f"  Disassembly available: {'Yes' if agent.state.get('disassembly') else 'No'}")
            print()
            continue

        elif user == "/verbose":
            agent.verbose = not agent.verbose
            verbose = agent.verbose
            if verbose:
                logger.setLevel(logging.DEBUG)
                agent.log.setLevel(logging.DEBUG)
                print("Verbose mode: ON")
            else:
                logger.setLevel(logging.INFO)
                agent.log.setLevel(logging.INFO)
                print("Verbose mode: OFF")
            continue

        elif user == "/log":
            file_logging_enabled = not file_logging_enabled
            if file_logging_enabled:
                print(f"File logging: ON (writing to {log_file})")
            else:
                print("File logging: OFF")
            # Toggle file handler
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    if not file_logging_enabled:
                        logger.removeHandler(handler)
                    break
            else:
                if file_logging_enabled and log_file:
                    file_handler = logging.FileHandler(log_file, mode='a')
                    file_handler.setFormatter(logging.Formatter(
                        '%(asctime)s - [%(levelname)s] %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S'
                    ))
                    logger.addHandler(file_handler)
            continue

        elif user == "/reset":
            messages = [{"role": "system", "content": system_prompt}]
            agent.state = {"binary": None, "functions": None, "disassembly": None}
            agent._boundary_loaded = False
            print("Conversation and state cleared.\n")
            logger.info("[RESET] Conversation and state cleared")
            continue

        elif user.startswith("/ask "):
            # Parse /ask command: /ask <binary_path> <question>
            parts = user[5:].strip().split(None, 1)  # Split on first space
            if len(parts) < 2:
                print("Usage: /ask <binary_path> <question>")
                print("Example: /ask /path/to/binary.bin Show the first 10 functions with comments")
                continue

            binary_path = parts[0]
            question = parts[1]

            # Create a comprehensive prompt that guides the model
            user_msg = f"""Analyze the binary file at: {binary_path}

To answer this question: {question}

Please follow this exact sequence:
1. First, call binary_loader with file_path="{binary_path}" to load the binary
2. After that succeeds, call boundary_detector to find function boundaries
3. Then call disassembler to get the assembly code
4. Finally, analyze the results to answer: {question}

Remember to use all three tools in sequence."""

            logger.info(f"[ASK COMMAND] Binary: {binary_path}, Question: {question}")

        elif user.startswith("/open "):
            path = user[6:].strip()
            if not path:
                print("Usage: /open <path>")
                continue

            # Simple analysis prompt for /open
            user_msg = f"""Analyze the binary file at: {path}

Please follow this exact sequence:
1. First, call binary_loader with file_path="{path}" to load the binary
2. After that succeeds, call boundary_detector to find function boundaries
3. Then call disassembler to get the assembly code
4. Finally, provide a summary of what this binary does based on the analysis

Use all three tools in sequence to perform a complete analysis."""

            logger.info(f"[OPEN COMMAND] Starting analysis of: {path}")

        else:
            # Regular message
            user_msg = user

        messages.append({"role": "user", "content": user_msg})
        logger.debug(f"[CONVERSATION] Added user message: {user_msg[:100]}...")

        # Tool loop for this turn
        iteration = 0
        max_iterations = 10  # Prevent infinite loops

        while iteration < max_iterations:
            iteration += 1
            logger.debug(f"[CHAT LOOP] Iteration {iteration}")

            try:
                # Make the API call
                logger.debug(f"[API] Calling {model_id} with {len(messages)} messages")
                resp = agent.client.chat.completions.create(
                    model=agent.model_id,
                    messages=messages,
                    functions=agent.functions,
                    function_call="auto",
                    temperature=0.3,  # Lower temperature for more consistent tool use
                )
                msg = resp.choices[0].message

            except Exception as e:
                error_msg = f"Error communicating with model: {e}"
                print(f"\nagent> {error_msg}\n")
                logger.error(f"[API ERROR] {e}", exc_info=True)
                break

            # Log response
            if verbose:
                logger.debug(f"[LLM RESPONSE] Content: {msg.content}")
                if hasattr(msg, 'function_call') and msg.function_call:
                    logger.debug(f"[LLM RESPONSE] Function call: {msg.function_call.name}")
                    logger.debug(f"[LLM RESPONSE] Function args: {msg.function_call.arguments}")

            # Check if model wants to call a tool
            if getattr(msg, "function_call", None):
                name = msg.function_call.name
                raw = msg.function_call.arguments or "{}"

                print(f"[Calling tool: {name}]")
                logger.info(f"[LLM DECISION] Model wants to call tool: {name}")
                logger.debug(f"[LLM DECISION] Tool arguments: {raw}")

                try:
                    parsed = json.loads(raw)
                except Exception as e:
                    logger.error(f"[ERROR] Failed to parse tool arguments: {e}")
                    parsed = {}

                # Execute the tool
                func_tuple = agent._tools.get(name)
                if not func_tuple:
                    content = json.dumps({"error": f"Unknown tool: {name}"})
                    logger.error(f"[ERROR] Unknown tool requested: {name}")
                else:
                    func, ArgsModel = func_tuple
                    kwargs, err = agent._validate_args(parsed, ArgsModel)
                    if err:
                        content = json.dumps({"error": f"Invalid arguments: {err}"})
                        logger.error(f"[ERROR] Invalid tool arguments: {err}")
                    else:
                        logger.info(f"[TOOL EXEC] Executing {name}")
                        content = func(**kwargs)
                        logger.info(f"[TOOL RESULT] {name} returned: {content[:200]}...")

                        # Show tool result to user in a nice format
                        try:
                            result = json.loads(content)
                            if "error" in result:
                                print(f"  ❌ {result['error']}")
                            else:
                                print(f"  ✓ {result.get('summary', 'Success')}")
                        except:
                            print(f"  Result: {content[:100]}...")

                messages.append({"role": "function", "name": name, "content": content})
                continue

            # Normal assistant reply
            text = (msg.content or "").strip()
            if text:
                print(f"\nagent> {text}\n")
                logger.info(f"[ASSISTANT RESPONSE]\n{text}")
            messages.append({"role": "assistant", "content": text})
            break

    logger.info("="*80)
    logger.info(f"CHAT SESSION ENDED - {datetime.now()}")
    logger.info("="*80)


def main():
    ap = argparse.ArgumentParser(description="Chat with the GPT-OSS binary analysis agent.")
    ap.add_argument("--binary", help="Default binary path for analysis")
    ap.add_argument("--boundary-ckpt", help="Path to boundary detector model checkpoint")
    ap.add_argument("--model-id", default="gpt-oss:20b", help="Model ID for Ollama")
    ap.add_argument("--ollama-url", default="http://localhost:11434/v1", help="Ollama API URL")
    ap.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    ap.add_argument("--log-file", help="Custom log file path")
    ap.add_argument("--log-dir", default="./logs", help="Log directory")
    ap.add_argument("--no-console", action="store_true", help="Disable console output")
    ap.add_argument("--enhanced", action="store_true", help="Use enhanced prompts for better tool use")

    args = ap.parse_args()

    log_file = setup_log_file(args)

    if not args.no_console:
        print("="*60)
        print("GPT-OSS Binary Analysis Chat")
        print(f"Model: {args.model_id}")
        print(f"Log: {log_file}")
        print(f"Enhanced prompts: {args.enhanced}")
        print("="*60)
        print()

    run_chat(
        args.binary,
        args.boundary_ckpt,
        args.model_id,
        args.ollama_url,
        args.verbose,
        log_file,
        args.no_console,
        args.enhanced
    )

    if not args.no_console:
        print(f"\nSession ended. Log: {log_file}")


if __name__ == "__main__":
    sys.exit(main())
