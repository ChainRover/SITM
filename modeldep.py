import argparse
import json
import os
import re
import time
from pathlib import Path


DEFAULT_MODEL = ""
DEFAULT_BASE_URL = "http://localhost:11434/v1"


def remove_code_fence(text):
    text = text.strip()
    text = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\s*\n?", "", text)
    return re.sub(r"\s*```\s*$", "", text).strip()


def remove_think(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE).strip()


def repair_incomplete_json(text):
    """Read a JSON object, allowing fences and missing outer closing brackets.

    An unfinished string is left as a failure: its missing source cannot be recovered.
    """
    text = remove_code_fence(remove_think(text or ""))
    start = text.find("{")
    if start < 0:
        return None
    text = text[start:]
    try:
        result, end = json.JSONDecoder().raw_decode(text)
        # print("JSON consumed chars:", end, "response chars:", len(text))
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass

    stack = []
    quoted = False
    escaped = False
    for character in text:
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character in "{[":
            stack.append("}" if character == "{" else "]")
        elif character in "}]":
            if not stack or stack.pop() != character:
                return None
    # print("unfinished JSON:", "inside string:", quoted, "closing brackets:", stack)
    if quoted:
        return None
    try:
        result = json.loads(text + "".join(reversed(stack)))
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        return None


def extract_solidity_from_answer(response):
    text = remove_think(response or "")
    if not text:
        return ""
    if text.startswith("{") or text.startswith("```json"):
        result = repair_incomplete_json(text)
        if result is not None:
            source = result.get("source_code", "")
            if not isinstance(source, str) or result.get("success", 1) != 1:
                return ""
            text = source.strip()
    answer = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer:
        text = answer[1]
    else:
        answer = re.search(r"<answer>\s*(.*)", text, re.DOTALL | re.IGNORECASE)
        if answer:
            text = answer[1]
        else:
            block = re.search(r"```(?:solidity|sol|javascript|js)?[ \t]*\n?(.*?)```",
                              text, re.DOTALL | re.IGNORECASE)
            if block:
                text = block[1]
    source = remove_code_fence(text)
    # print("extracted source chars:", len(source), "head:", source[:80])
    if not re.search(r"\b(?:pragma\s+solidity|contract\s+\w+|library\s+\w+|interface\s+\w+)", source):
        return ""
    return source


def output_format_prompt(sol_v, output_format="xml"):
    rules = f"""Use Solidity version {sol_v}. Return complete source that can be compiled with that version.
Preserve the observable behavior, storage layout, external calls, events, and error handling.
Use meaningful placeholder names when original names cannot be recovered from bytecode.
Keep syntax compatible with the specified compiler. Do not introduce unrelated events or imports."""
    if output_format == "xml":
        return rules + "\nPut the source inside <answer>...</answer>. Do not output JSON or Markdown fences."
    if output_format != "json":
        raise ValueError("Output format must be xml or json")
    fields = {
        "success": 1, "solidity_version": sol_v, "contract_name": "UnknownContract",
        "source_code": "Complete Solidity source", "decompilation_notes": "Uncertainties and assumptions",
        "confidence": 0, "warnings": [], "bytecode_hash": "",
    }
    return (rules + "\nReturn only a JSON object with the following fields. success is 0 or 1; "
            "confidence is an integer from 0 to 100. If generation fails, use success=0 and source_code=\"\". "
            "Leave bytecode_hash empty unless it was actually computed.\n" + json.dumps(fields, ensure_ascii=False))


def build_bytecode_to_solidity_prompt(bytecode, sol_v, output_format="xml"):
    return ("Reconstruct readable Solidity source from the supplied Ethereum runtime bytecode.\n\n"
            + output_format_prompt(sol_v, output_format)
            + f"\n\nCompiler version hint:\n{sol_v}\n\nRuntime bytecode:\n{bytecode}")


def build_tac_to_solidity_prompt(tac, sol_v, output_format="xml"):
    return ("Reconstruct readable Solidity source from Gigahorse three-address code (TAC). "
            "Use its function boundaries, storage operations, and control flow.\n\n"
            + output_format_prompt(sol_v, output_format)
            + f"\n\nCompiler version hint:\n{sol_v}\n\nTAC:\n{tac}")


# 保留原脚本的函数名，已有调用可以继续使用。
def build_TAC_to_solidity_prompt(tac, sol_v, output_format="xml"):
    return build_tac_to_solidity_prompt(tac, sol_v, output_format)


def build_distillation_prompt(tac_code, solidity_code):
    return f"""Explain the correspondence between the supplied TAC and reference Solidity.
Cover storage slots and state variables, function signatures and argument decoding,
control-flow reconstruction, and compiler-generated checks. Describe concrete evidence
and mark uncertain mappings. Return only a JSON object with a string field named "think".

TAC:
{tac_code}

Reference Solidity:
{solidity_code}"""


class BuildAgent:
    def __init__(self, model="", base_url=None, api_key=None, timeout=300, temperature=0.1, top_p=0.9):
        from openai import OpenAI

        self.model = model or os.environ.get("MODEL_NAME") or DEFAULT_MODEL
        self.base_url = base_url or os.environ.get("MODEL_BASE_URL") or DEFAULT_BASE_URL
        key = api_key or os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            from urllib.parse import urlparse

            if urlparse(self.base_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
                raise ValueError("Set MODEL_API_KEY or pass api_key for the configured model endpoint")
            key = "ollama"
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        self.timeout = timeout
        self.temperature = temperature
        self.top_p = top_p
        # 重试由下面的循环控制，避免和 SDK 的自动重试叠加。
        self.client = OpenAI(api_key=key, base_url=self.base_url, timeout=timeout, max_retries=0)
        self.last_response = ""
        self.last_error = ""
        print(f"Model: {self.model}; endpoint: {self.base_url}")

    def output_format_prompt(self, sol_v):
        return output_format_prompt(sol_v, "json")

    def build_bytecode_to_solidity_prompt(self, bytecode, sol_v):
        return build_bytecode_to_solidity_prompt(bytecode, sol_v, "json")

    def build_TAC_to_solidity_prompt(self, tac, sol_v):
        return build_tac_to_solidity_prompt(tac, sol_v, "json")

    def build_distillation_prompt(self, tac_code, solidity_code):
        return build_distillation_prompt(tac_code, solidity_code)

    def answer(self, prompt, timeout=None):
        # print("prompt chars:", len(prompt), "model:", self.model)
        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], stream=False,
            temperature=self.temperature, top_p=self.top_p,
            timeout=self.timeout if timeout is None else timeout,
        )
        if not response.choices:
            raise ValueError("The model returned no choices")
        self.last_response = response.choices[0].message.content or ""
        # print("response chars:", len(self.last_response), "finish reason:", response.choices[0].finish_reason)
        if response.choices[0].finish_reason == "length":
            raise ValueError("Model output was truncated; the generated source is incomplete")
        if not self.last_response.strip():
            raise ValueError("The model returned empty content")
        return self.last_response

    def call_with_retry(self, prompt, mode, max_retries, base_delay):
        if max_retries < 1 or base_delay < 0:
            raise ValueError("max_retries must be positive and base_delay nonnegative")
        self.last_error = ""
        self.last_response = ""
        for attempt in range(max_retries):
            try:
                self.last_response = ""
                text = self.answer(prompt)
                if mode == "xml":
                    source = extract_solidity_from_answer(text)
                    if not source:
                        raise ValueError("No Solidity source found in the response")
                    self.last_error = ""
                    return {"success": 1, "source_code": source}
                result = repair_incomplete_json(text)
                if result is None:
                    raise ValueError("The model response is not a complete JSON object")
                if mode == "think":
                    if not isinstance(result.get("think"), str) or not result["think"].strip():
                        raise ValueError("The response needs a nonempty think field")
                elif mode == "json":
                    if result.get("success") not in (0, 1) or not isinstance(result.get("source_code"), str):
                        raise ValueError("The response needs success=0/1 and a source_code string")
                    if result["success"] == 1:
                        source = extract_solidity_from_answer(result["source_code"])
                        if not source:
                            raise ValueError("success=1 but no Solidity source was provided")
                        result["source_code"] = source
                else:
                    raise ValueError(f"Unknown response mode: {mode}")
                self.last_error = ""
                return result
            except Exception as error:
                self.last_error = f"{type(error).__name__}: {error}"
                print(f"Attempt {attempt + 1}/{max_retries} failed: {self.last_error}", flush=True)
                if getattr(error, "status_code", None) in {400, 401, 403, 404}:
                    break
                if attempt + 1 < max_retries:
                    delay = base_delay * (attempt + 1)
                    # print("retry delay seconds:", delay)
                    time.sleep(delay)
        return None

    def repeat_call(self, prompt, max_retries=3, base_delay=2):
        result = self.call_with_retry(prompt, "xml", max_retries, base_delay)
        return result if result is not None else {"success": 0, "source_code": "", "error": self.last_error}

    def repeat_call_json(self, prompt, max_retries=3, base_delay=2):
        return self.call_with_retry(prompt, "json", max_retries, base_delay)

    def generate_think(self, tac_code, solidity_code, max_retries=3, base_delay=2):
        prompt = self.build_distillation_prompt(tac_code, solidity_code)
        return self.call_with_retry(prompt, "think", max_retries, base_delay)

    def close(self):
        self.client.close()


def run(args):
    input_path = args.bytecode_file or args.tac_file
    content = input_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        raise ValueError(f"Empty input file: {input_path}")
    if args.solidity_file and not args.tac_file:
        raise ValueError("--solidity-file is used with --tac-file for reference explanations")
    if args.bytecode_file:
        content = "0x" + bytes.fromhex(content.removeprefix("0x").removeprefix("0X")).hex()
    agent = BuildAgent(args.model, args.base_url, timeout=args.timeout,
                       temperature=args.temperature, top_p=args.top_p)
    try:
        if args.solidity_file:
            reference = args.solidity_file.read_text(encoding="utf-8-sig")
            result = agent.generate_think(content, reference, args.retries, args.delay)
            source = None
        else:
            builder = build_bytecode_to_solidity_prompt if args.bytecode_file else build_tac_to_solidity_prompt
            prompt = builder(content, args.sol_version, args.format)
            # print("input file:", input_path, "format:", args.format, "prompt chars:", len(prompt))
            if args.format == "json":
                result = agent.repeat_call_json(prompt, args.retries, args.delay)
            else:
                result = agent.repeat_call(prompt, args.retries, args.delay)
            source = result.get("source_code", "") if result and result.get("success") == 1 else ""
        if args.raw_output:
            args.raw_output.parent.mkdir(parents=True, exist_ok=True)
            args.raw_output.write_text(agent.last_response, encoding="utf-8")
        if result is None or (source is not None and not source):
            raise ValueError(agent.last_error or "The model reported generation failure")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        saved = json.dumps(result, ensure_ascii=False, indent=2) if source is None else source
        args.output.write_text(saved.rstrip() + "\n", encoding="utf-8")
        print(f"Saved {args.output}")
        if source is not None:
            print("Source extracted; compilation and behavioral equivalence have not been checked")
    finally:
        agent.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--bytecode-file", type=Path)
    inputs.add_argument("--tac-file", type=Path)
    parser.add_argument("--solidity-file", type=Path, help="Reference source for TAC explanations")
    parser.add_argument("--sol-version", default="0.8.20")
    parser.add_argument("--model", default="")
    parser.add_argument("--base-url")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--format", choices=("xml", "json"), default="xml")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay", type=float, default=2)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-output", type=Path)
    args = parser.parse_args()
    if args.timeout <= 0 or args.retries < 1 or args.delay < 0:
        parser.error("Timeout/retries must be positive; delay must be nonnegative")
    if args.raw_output and args.raw_output.resolve() == args.output.resolve():
        parser.error("--output and --raw-output need different paths")
    input_paths = {path.resolve() for path in (args.bytecode_file, args.tac_file, args.solidity_file) if path}
    if any(path and path.resolve() in input_paths for path in (args.output, args.raw_output)):
        parser.error("Output paths must differ from input files")
    try:
        run(args)
    except ImportError as error:
        parser.exit(2, f"Missing dependency: {error}. Install openai with python -m pip install openai.\n")
    except Exception as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    main()
