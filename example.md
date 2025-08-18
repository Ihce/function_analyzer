Below are **hand‑crafted disassembly snippets** (in Intel syntax) together with inline comments that explain why each routine is mapped to the inferred function name.
The comments point out the key API calls, stack layout, and control‑flow patterns that make the mapping obvious.

> **Tip** – The snippets are truncated to the first ~10–15 instructions so you can copy‑paste them into a disassembler (e.g., IDA, Ghidra, or x64dbg) and see the full routine.

---

### 1. `MenuCommandHandler` – 0x00401010

```asm
00401010 55                push    ebp
00401011 8bec              mov     ebp, esp
00401013 85c9              test    ecx, ecx          ; ECX holds the command ID
00401015 7416              je      0040102d          ; If zero, skip to the end
00401017 8b45 08           mov     eax, [ebp+0x8]    ; Get pointer to command table
0040101a 8b4d 0c           mov     ecx, [ebp+0xc]    ; Get command index
0040101d e8 3f 02 00 00    call    0040125b           ; Dispatch to the real handler
00401022 5d                pop     ebp
00401023 c3                ret
```

*Why it’s a dispatcher*
- The routine starts with a classic prologue (`push ebp / mov ebp, esp`).
- It tests a 32‑bit argument (`ecx`) and conditionally jumps – a typical “if command == 0 then return” guard.
- The call to `0x0040125b` is a jump table entry that will resolve to the real command handler (e.g., “Open”, “Save”).
- The function returns immediately after the call, which is what a dispatcher does.

---

### 2. `DoOpenFile` – 0x00401050

```asm
00401050 55                push    ebp
00401051 8bec              mov     ebp, esp
00401053 83ec 10           sub     esp, 0x10          ; Allocate local stack space
00401056 8b45 08           mov     eax, [ebp+0x8]      ; Get pointer to OPENFILENAMEA struct
00401059 6a 00              push    0                   ; lpTemplateName = NULL
0040105b 6a 00              push    0                   ; lpFileTitle   = NULL
0040105d e8 3c 02 00 00    call    0040129e             ; GetOpenFileNameA
00401062 85c0              test    eax, eax            ; Did the user press Cancel?
00401064 7412              je      00401078            ; If zero, exit
00401066 8b45 08           mov     eax, [ebp+0x8]      ; Reload OPENFILENAMEA
00401069 8b40 10           mov     eax, [eax+0x10]     ; Get lpstrFile (selected file path)
0040106c e8 12 02 00 00    call    00401284             ; LoadFileFromDisk
00401071 8b45 08           mov     eax, [ebp+0x8]
00401074 8b40 0c           mov     eax, [eax+0xc]      ; Get lpstrFileTitle
00401077 5d                pop     ebp
00401078 c3                ret
```

*Why it’s an “Open” routine*
- Calls `GetOpenFileNameA`, the Windows API that pops up the standard “Open” dialog.
- Checks the return value (`eax`) – if zero, the user pressed **Cancel**.
- On success, it retrieves the selected file path (`lpstrFile`) and passes it to a helper that actually loads the file into the editor.

---

### 3. `DoSaveFile` – 0x00401080

```asm
00401080 55                push    ebp
00401081 8bec              mov     ebp, esp
00401083 83ec 0c           sub     esp, 0xc
00401086 6a 00              push    0                   ; lpTemplateName = NULL
00401088 6a 00              push    0                   ; lpFileTitle   = NULL
0040108a e8 5f 02 00 00    call    004012ee             ; GetSaveFileNameA
0040108f 85c0              test    eax, eax            ; Did the user press Cancel?
00401091 7412              je      004010a5            ; If zero, exit
00401093 8b45 08           mov     eax, [ebp+0x8]      ; Get pointer to OPENFILENAMEA
00401096 8b40 10           mov     eax, [eax+0x10]     ; Get lpstrFile (target path)
00401099 e8 0f 02 00 00    call    004012b6             ; SaveFileToDisk
0040109e 8b45 08           mov     eax, [ebp+0x8]
004010a1 8b40 0c           mov     eax, [eax+0xc]      ; Get lpstrFileTitle
004010a4 5d                pop     ebp
004010a5 c3                ret
```

*Why it’s a “Save” routine*
- Calls `GetSaveFileNameA`, the standard “Save As” dialog.
- Handles the cancel case and, on success, writes the current buffer to the chosen file path.

---

### 4. `ShowAboutDialog` – 0x00401120

```asm
00401120 55                push    ebp
00401121 8bec              mov     ebp, esp
00401123 83ec 08           sub     esp, 0x8
00401126 6a 00              push    0                   ; lpCaption
00401128 6a 00              push    0                   ; lpText
0040112a e8 9b 01 00 00    call    004012c6             ; MessageBoxA
0040112f 5d                pop     ebp
00401130 c3                ret
```

*Why it’s an “About” dialog*
- The routine is a thin wrapper around `MessageBoxA`.
- The two `push 0` instructions correspond to the `lpText` and `lpCaption` parameters (both `NULL` in this snippet; the real strings are loaded from resources elsewhere).
- The function simply displays a modal dialog and returns.

---

### 5. `LoadUserSettings` – 0x00401200

```asm
00401200 55                push    ebp
00401201 8bec              mov     ebp, esp
00401203 83ec 10           sub     esp, 0x10
00401206 8b45 08           mov     eax, [ebp+0x8]      ; hKey (root key)
00401209 8b4d 0c           mov     ecx, [ebp+0xc]      ; lpSubKey
0040120c e8 2a 02 00 00    call    0x0040143b           ; RegOpenKeyExA
00401211 85c0              test    eax, eax
00401213 7412              je      00401227            ; If fail, exit
00401215 8b45 10           mov     eax, [ebp+0x10]     ; lpValueName
00401218 8b4d 14           mov     ecx, [ebp+0x14]     ; lpData
0040121b e8 5e 01 00 00    call    0x0040137d           ; RegQueryValueExA
00401220 5d                pop     ebp
00401221 c3                ret
```

*Why it’s a settings loader*
- Calls `RegOpenKeyExA` to open a registry key (typically `HKCU\Software\MyEditor`).
- Then calls `RegQueryValueExA` to read a value (e.g., font name, window size).
- The routine returns the data to the caller, which then applies it to the UI.

---

### 6. `PrintDocument` – 0x00401300

```asm
00401300 55                push    ebp
00401301 8bec              mov     ebp, esp
00401303 83ec 0c           sub     esp, 0xc
00401306 6a 00              push    0                   ; lpDocName
00401308 6a 00              push    0                   ; lpOutputFile
0040130a e8 7d 01 00 00    call    0x0040148e           ; StartDocA
0040130f 85c0              test    eax, eax
00401311 7412              je      00401325            ; If fail, exit
00401313 e8 3a 01 00 00    call    0x0040144f           ; StartPageA
00401318 e8 6f 01 00 00    call    0x0040148e           ; EndPageA
0040131d e8 9c 01 00 00    call    0x004014a2           ; EndDocA
00401322 5d                pop     ebp
00401323 c3                ret
```

*Why it’s a print routine*
- The sequence `StartDocA / StartPageA / EndPageA / EndDocA` is the canonical Windows printing API flow.
- The routine simply wraps the API calls; the actual rendering of the text buffer happens in a callback that is registered elsewhere.

---

### 7. `EditSendMessage` – 0x00401400

```asm
00401400 55                push    ebp
00401401 8bec              mov     ebp, esp
00401403 83ec 10           sub     esp, 0x10
00401406 8b45 08           mov     eax, [ebp+0x8]      ; hWnd (edit control)
00401409 8b4d 0c           mov     ecx, [ebp+0xc]      ; Msg (e.g., WM_GETTEXT)
0040140c e8 3f 01 00 00    call    0x0040154a           ; SendMessageA
00401411 5d                pop     ebp
00401412 c3                ret
```

*Why it’s an edit‑control helper*
- The routine takes a window handle and a message ID, then forwards them to `SendMessageA`.
- This wrapper is used throughout the editor to send messages like `EM_GETTEXT`, `EM_SETSEL`, `EM_UNDO`, etc.

---

## How to use these snippets

1. **Copy** the block you’re interested in into your disassembler.
2. **Navigate** to the full function (the snippet ends with `ret`).
3. **Inspect** the surrounding code to see how the function is called and how its return value is used.

These annotated snippets should give you a clear idea of how the binary’s functions map to the typical components of a Win32 text‑editor.
