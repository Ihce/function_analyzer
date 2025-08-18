#!/usr/bin/env python3
import argparse
import json
import sys

from agent import BinaryAnalysisAgent


def run_chat(binary: str | None, boundary_ckpt: str | None, model_id: str, ollama_url: str):
    agent = BinaryAnalysisAgent(
        ollama_url=ollama_url,
        model_id=model_id,
        boundary_ckpt=boundary_ckpt,
    )

    # persistent chat history
    messages = [
        {"role": "system", "content": "You are a concise reverse-engineering assistant."}
    ]

    print("Chat started. Type your question, or use commands:")
    print("  /open <path>   — analyze this binary (adds a user message with the path)")
    print("  /reset         — clear conversation state (tools' cached state stays in memory)")
    print("  /exit          — quit\n")

    if binary:
        print(f"(hint) You passed --binary {binary}. Use /open {binary} to kick off analysis.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            messages = [{"role": "system", "content": "You are a concise reverse-engineering assistant."}]
            print("(reset) Conversation cleared.\n")
            continue
        if user.startswith("/open "):
            path = user[len("/open "):].strip()
            if not path:
                print("usage: /open <path>")
                continue
            # Let the model decide the tool sequence; we just ask it to analyze that path.
            user_msg = f"Analyze this binary and explain what it does: {path}"
        else:
            user_msg = user

        messages.append({"role": "user", "content": user_msg})

        # tool loop for this turn
        while True:
            resp = agent.client.chat.completions.create(
                model=agent.model_id,
                messages=messages,
                functions=agent.functions,
                function_call="auto",
            )
            msg = resp.choices[0].message

            # If the model wants to call a tool, execute it and feed back the result
            if getattr(msg, "function_call", None):
                name = msg.function_call.name
                raw = msg.function_call.arguments or "{}"
                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = {}

                # convenience: if it calls binary_loader without a path and you supplied --binary, fill it
                if name == "binary_loader" and "file_path" not in parsed and binary:
                    parsed["file_path"] = binary

                func_tuple = agent._tools.get(name)  # (callable, ArgsModel)
                if not func_tuple:
                    content = json.dumps({"error": f"Unknown tool: {name}"})
                else:
                    func, ArgsModel = func_tuple
                    kwargs, err = agent._validate_args(parsed, ArgsModel)
                    if err:
                        content = json.dumps({"error": f"Invalid arguments: {err}"})
                    else:
                        content = func(**kwargs)  # calls agent’s wrapper (which uses your components)

                messages.append({"role": "function", "name": name, "content": content})
                continue

            # Otherwise it's a normal assistant reply; show it and end this turn
            text = (msg.content or "").strip()
            if text:
                print(f"\nagent> {text}\n")
            messages.append({"role": "assistant", "content": text})
            break


def main():
    ap = argparse.ArgumentParser(description="Chat with the GPT-OSS binary analysis agent.")
    ap.add_argument("--binary", help="Optional default binary path to analyze (used if model omits path).")
    ap.add_argument("--boundary-ckpt", help="Path to boundary detector model checkpoint (optional).")
    ap.add_argument("--model-id", default="gpt-oss:20b", help="Model id for Ollama.")
    ap.add_argument("--ollama-url", default="http://localhost:11434/v1", help="OpenAI-compatible base URL.")
    args = ap.parse_args()
    run_chat(args.binary, args.boundary_ckpt, args.model_id, args.ollama_url)


if __name__ == "__main__":
    sys.exit(main())
