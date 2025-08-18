"""Tool for disassembling detected functions."""

import capstone
from typing import Dict, Any, List

# ✨ Pydantic (self-describing tool schema)
from pydantic import BaseModel


# ✨ Args model for function-calling (no args needed for this tool)
class DisassemblerArgs(BaseModel):
    # Intentionally empty — agent supplies state (text_array, functions, base_address, arch)
    pass


class DisassemblerTool:
    """Disassemble functions using Capstone."""

    name = "disassembler"
    description = """Disassembles binary functions into assembly instructions.
    Input: text_array, functions list, architecture
    Output: disassembled instructions for each function
    Use this after detecting function boundaries to see the actual assembly code."""

    # ✨ Tell the agent what args this tool expects
    ArgsModel = DisassemblerArgs

    @classmethod
    def tool_spec(cls) -> Dict[str, Any]:
        """Return OpenAI/Ollama function-calling schema generated from Pydantic."""
        try:
            params = cls.ArgsModel.model_json_schema()  # Pydantic v2
        except AttributeError:
            params = cls.ArgsModel.schema()             # Pydantic v1 fallback
        params.setdefault("additionalProperties", False)
        return {
            "name": cls.name,
            "description": cls.description.strip(),
            "parameters": params,
        }

    def __init__(self):
        self.cs_cache = {}

    def _get_capstone(self, arch: str, is_64bit: bool):
        """Get or create Capstone instance."""
        key = (arch, is_64bit)
        if key not in self.cs_cache:
            if is_64bit:
                cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
            else:
                cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
            cs.detail = True
            self.cs_cache[key] = cs
        return self.cs_cache[key]

    def run(self, text_array: bytes, functions: List[Dict],
            base_address: int, architecture: str = "x64") -> Dict[str, Any]:
        """Execute the tool."""
        try:
            is_64bit = "64" in architecture
            cs = self._get_capstone(architecture, is_64bit)

            disassembled_functions = []

            for func in functions:
                start = func["start_offset"]
                end = func["end_offset"] + 1

                func_bytes = text_array[start:end]
                func_address = base_address + start

                instructions = []
                for insn in cs.disasm(func_bytes, func_address):
                    instructions.append({
                        "address": f"0x{insn.address:08x}",
                        "mnemonic": insn.mnemonic,
                        "op_str": insn.op_str,
                        "bytes": insn.bytes.hex()
                    })

                # Format as text
                disasm_text = []
                for insn in instructions[:20]:  # First 20 instructions
                    disasm_text.append(
                        f"{insn['address']}:  {insn['bytes']:20s}  {insn['mnemonic']} {insn['op_str']}"
                    )

                disassembled_functions.append({
                    "function": func,
                    "instruction_count": len(instructions),
                    "disassembly_preview": "\n".join(disasm_text),
                    "instructions": instructions
                })

            return {
                "success": True,
                "disassembled_functions": disassembled_functions,
                "total_functions": len(disassembled_functions),
                "message": f"Successfully disassembled {len(disassembled_functions)} functions"
            }

        except Exception as e:
            return {"error": f"Disassembly failed: {str(e)}", "success": False}
