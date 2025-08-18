#!/usr/bin/env python3
import argparse
import json
import sys
import os
from datetime import datetime
import logging
from .agent import BinaryAnalysisAgent


# Enhanced system prompt that explicitly mentions tools
SYSTEM_PROMPT = """You are a reverse-engineering assistant with access to specialized tools for binary analysis.

You have access to the following tools:
1. binary_loader - Loads PE binary files and extracts the .text section
2. boundary_detector - Uses neural networks to detect function boundaries in binary code
3. disassembler - Disassembles binary functions into readable assembly code

When asked about a binary file or to analyze code:
- First use binary_loader to load the file
- Then use boundary_detector to find functions
- Finally use disassembler to get the assembly code
- Analyze the results and provide insights

Always use tools when asked about binary files, assembly code, or reverse engineering tasks.
Be concise but thorough in your analysis."""


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


def create_enhanced_tool_descriptions():
    """Create more detailed tool descriptions to help the LLM understand when to use them."""
    return [
        {
            "name": "binary_loader",
            "description": "ALWAYS use this FIRST when analyzing any binary file. Loads a PE binary file and extracts the .text section containing executable code. Returns the binary data, architecture (x86/x64), and base address.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the PE binary file to load"
                    }
                },
                "required": ["file_path"]
            }
        },
        {
            "name": "boundary_detector",
            "description": "Use this AFTER binary_loader to find where functions start and end in the binary. Uses a neural network to detect function boundaries. Returns a list of functions with their addresses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_path": {
                        "type": "string",
                        "description": "Optional path to the boundary detection model (leave empty to use default)"
                    }
                },
                "required": []
            }
        },
        {
            "name": "disassembler",
            "description": "Use this AFTER boundary_detector to convert binary code to readable assembly. Disassembles the detected functions into assembly instructions. Returns human-readable assembly code.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    ]


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

    # Use enhanced tool descriptions if requested
    if enhanced_prompts:
        agent.functions = create_enhanced_tool_descriptions()
        system_prompt = SYSTEM_PROMPT
        logging.info("Using enhanced prompts and tool descriptions")
    else:
        system_prompt = "You are a concise reverse-engineering assistant."

    logging.info("="*80)
    logging.info(f"CHAT SESSION STARTED - {datetime.now()}")
    logging.info(f"Model: {model_id}")
    logging.info(f"Binary hint: {binary if binary else 'None provided'}")
    logging.info(f"Enhanced prompts: {enhanced_prompts}")
    logging.info("="*80)

    messages = [{"role": "system", "content": system_prompt}]

    print(f"Chat started. Log file: {log_file}")
    print("\nCommands:")
    print("  /help          — show this help")
    print("  /open <path>   — analyze a binary file")
    print("  /ask <query>   — ask about binary analysis (guides tool use)")
    print("  /tools         — list available tools")
    print("  /state         — show current analysis state")
    print("  /reset         — clear conversation and state")
    print("  /verbose       — toggle verbose mode")
    print("  /exit          — quit\n")

    if binary:
        print(f"Hint: You have --binary {binary}. Try asking questions like:")
        print(f"  'What does {binary} do?'")
        print(f"  'Analyze {binary} and find the main function'")
        print(f"  'Show me the assembly code from {binary}'\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            logging.info("Chat session ended by user interrupt")
            break

        if not user:
            continue

        logging.info(f"[USER INPUT] {user}")

        # Handle commands
        if user == "/exit":
            logging.info("Chat session ended by /exit command")
            break

        elif user == "/help":
            print("\nCommands:")
            print("  /open <path>   — analyze a binary file")
            print("  /ask <query>   — ask about binary analysis")
            print("  /tools         — list available tools")
            print("  /state         — show current analysis state")
            print("  /reset         — clear conversation and state")
            print("  /verbose       — toggle verbose mode")
            print("  /exit          — quit\n")
            continue

        elif user == "/tools":
            print("\nAvailable tools:")
            for i, tool in enumerate(agent.functions, 1):
                print(f"{i}. {tool['name']}: {tool['description'][:100]}...")
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
            if agent.verbose:
                logging.setLevel(logging.DEBUG)
                print("Verbose mode: ON")
            else:
                logging.setLevel(logging.INFO)
                print("Verbose mode: OFF")
            continue

        elif user == "/reset":
            messages = [{"role": "system", "content": system_prompt}]
            agent.state = {"binary": None, "functions": None, "disassembly": None}
            agent._boundary_loaded = False
            print("Conversation and state cleared.\n")
            logging.info("[RESET] Conversation and state cleared")
            continue

        elif user.startswith("/open "):
            path = user[len("/open "):].strip()
            if not path:
                print("Usage: /open <path>")
                continue
            # Explicitly guide the model to analyze
            user_msg = f"Please analyze the binary file at {path}. First load it with binary_loader, then detect functions with boundary_detector, and finally disassemble with disassembler. Tell me what you find."
            logging.info(f"[OPEN COMMAND] Guided analysis of: {path}")

        elif user.startswith("/ask "):
            query = user[len("/ask "):].strip()
            if not query:
                print("Usage: /ask <query>")
                continue
            # Add context to help the model use tools
            if binary and "binary" not in query.lower() and "file" not in query.lower():
                user_msg = f"Regarding the binary file {binary}: {query}"
            else:
                user_msg = query
            logging.info(f"[ASK COMMAND] Query: {query}")

        else:
            # Regular message - check if it mentions binary analysis
            if binary and any(word in user.lower() for word in ['analyze', 'what', 'show', 'find', 'extract', 'reverse']):
                # Add context about the binary
                user_msg = f"{user} (Note: you can analyze the binary at {binary} using your tools)"
            else:
                user_msg = user

        messages.append({"role": "user", "content": user_msg})
        logging.debug(f"[CONVERSATION] Added user message: {user_msg[:100]}...")

        # Tool loop for this turn
        iteration = 0
        max_iterations = 10  # Prevent infinite loops

        while iteration < max_iterations:
            iteration += 1
            logging.debug(f"[CHAT LOOP] Iteration {iteration}")

            try:
                # Make the API call
                logging.debug(f"[API] Calling {model_id} with {len(messages)} messages")
                resp = agent.client.chat.completions.create(
                    model=agent.model_id,
                    messages=messages,
                    functions=agent.functions,
                    function_call="auto",
                    temperature=0.7,  # Add some randomness for better tool use
                )
                msg = resp.choices[0].message

            except Exception as e:
                error_msg = f"Error communicating with model: {e}"
                print(f"\nagent> {error_msg}\n")
                logging.error(f"[API ERROR] {e}", exc_info=True)
                break

            # Log response
            if agent.verbose:
                logging.debug(f"[LLM RESPONSE] Content: {msg.content}")
                if hasattr(msg, 'function_call') and msg.function_call:
                    logging.debug(f"[LLM RESPONSE] Function call: {msg.function_call.name}")
                    logging.debug(f"[LLM RESPONSE] Function args: {msg.function_call.arguments}")

            # Check if model wants to call a tool
            if getattr(msg, "function_call", None):
                name = msg.function_call.name
                raw = msg.function_call.arguments or "{}"

                print(f"[Calling tool: {name}]")
                logging.info(f"[LLM DECISION] Model wants to call tool: {name}")
                logging.debug(f"[LLM DECISION] Tool arguments: {raw}")

                try:
                    parsed = json.loads(raw)
                except Exception as e:
                    logging.error(f"[ERROR] Failed to parse tool arguments: {e}")
                    parsed = {}

                # Auto-fill for binary_loader
                if name == "binary_loader" and "file_path" not in parsed and binary:
                    logging.debug(f"[AUTO-FILL] Adding file_path={binary} to binary_loader")
                    parsed["file_path"] = binary

                # Execute the tool
                func_tuple = agent._tools.get(name)
                if not func_tuple:
                    content = json.dumps({"error": f"Unknown tool: {name}"})
                    logging.error(f"[ERROR] Unknown tool requested: {name}")
                else:
                    func, ArgsModel = func_tuple
                    kwargs, err = agent._validate_args(parsed, ArgsModel)
                    if err:
                        content = json.dumps({"error": f"Invalid arguments: {err}"})
                        logging.error(f"[ERROR] Invalid tool arguments: {err}")
                    else:
                        logging.info(f"[TOOL EXEC] Executing {name}")
                        content = func(**kwargs)
                        logging.info(f"[TOOL RESULT] {name} returned: {content[:200]}...")

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
                logging.info(f"[ASSISTANT RESPONSE]\n{text}")
            messages.append({"role": "assistant", "content": text})
            break

    logging.info("="*80)
    logging.info(f"CHAT SESSION ENDED - {datetime.now()}")
    logging.info("="*80)


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
