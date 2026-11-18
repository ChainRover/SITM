import argparse
import pickle
import runpy
import sys
from functools import lru_cache
from pathlib import Path

from model_scores import bytecode_id, export_bytecodes, score_file


def read_functions(path):
    # This is the format written by the official 2.Preprocess.py script.
    with path.open("rb") as stream:
        data = pickle.load(stream)
    return {Path(name).stem: functions for name, functions in data.items()}


def smarttrans_tokens(text, tokenizer, destination_id):
    encoded = tokenizer(text, truncation=True, max_length=512, padding="max_length")
    tokens = encoded["input_ids"]
    destinations = [index for index, token in enumerate(tokens) if token == destination_id]
    # print("JUMPDEST token:", destination_id, "positions:", destinations[:20])
    # print("function input tokens:", sum(encoded["attention_mask"]))
    # Same jump-token redirection as the official embedding script.
    for index, token in enumerate(tokens):
        if token < 4096 and token != index:
            # print("redirect jump token:", index, token, "target:", destinations[token] if token < len(destinations) else index)
            tokens[index] = destinations[token] if token < len(destinations) else index
    return tokens, encoded["attention_mask"]


def contract_similarity(left, right):
    # Both matrices hold unit-normalized function vectors.
    similarities = left @ right.T
    # print("function vectors:", tuple(left.shape), tuple(right.shape))
    # print("cosine range:", float(similarities.min()), float(similarities.max()))
    # print("left function matches:", similarities.argmax(dim=1).tolist())
    # print("函数匹配矩阵", similarities.shape)
    # print("左侧最佳匹配", similarities.max(dim=1).values)
    # print("右侧最佳匹配", similarities.max(dim=0).values)
    return float((similarities.max(dim=0).values.mean() + similarities.max(dim=1).values.mean()) / 2)


# 单独测试函数顺序对合约聚合的影响，两个矩阵都使用单位向量。
# import torch
# functions = torch.eye(3)
# print(contract_similarity(functions, functions))
# print(contract_similarity(functions, functions.flip(0)))


def score(args):
    import torch
    from transformers import AutoTokenizer

    for path in (args.repo / "models.py", args.pretrain / "config.json", args.checkpoint):
        if not path.is_file():
            raise ValueError(f"Missing required SmartTrans file: {path}")
    with_repo = str(args.repo.resolve())
    sys.path.insert(0, with_repo)
    try:
        JtransEncoder = runpy.run_path(str(args.repo / "models.py"))["JtransEncoder"]
        if args.facts:
            analyze = runpy.run_path(str(args.repo / "1.Analyze_gigahorse.py"))["extract_cfg"]
            analyze.__globals__["TEMP_PATH"] = str(args.facts.resolve())
            preprocess = runpy.run_path(str(args.repo / "2.Preprocess.py"))["preprocess"]

            def functions_for(name):
                graph = analyze(name)
                # print("Gigahorse bytecode:", name, "graph recovered:", bool(graph))
                if not graph or bytecode_id(graph["bytecode"]) != name:
                    raise ValueError(f"Gigahorse facts missing or bytecode mismatch: {name}")
                return preprocess(graph)
        else:
            functions = read_functions(args.functions)

            def functions_for(name):
                if name not in functions:
                    raise ValueError(f"No SmartTrans functions for {name}")
                return functions[name]

        tokenizer = AutoTokenizer.from_pretrained(args.repo / "tokenizer", local_files_only=True)
        destination_id = tokenizer("JUMPDEST")["input_ids"][0]
        model = JtransEncoder(dim=128, pretrain_path=str(args.pretrain.resolve()))
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        # print("checkpoint tensors:", len(state), "device:", args.device)
        model.to(args.device).eval()

        @lru_cache(maxsize=256)
        def embed(name):
            functions = functions_for(name)
            if not functions:
                raise ValueError(f"No functions recovered for {name}")
            texts = list(functions.values())
            # print("contract functions:", name, len(texts))
            # print("first function tokens:", texts[0].split()[:30])
            vectors = []
            with torch.inference_mode():
                for start in range(0, len(texts), args.batch_size):
                    batch = [smarttrans_tokens(text, tokenizer, destination_id)
                             for text in texts[start:start + args.batch_size]]
                    ids, masks = zip(*batch)
                    # print("function batch:", start, "size:", len(ids), "sequence length:", len(ids[0]))
                    
                    outputs = model(input_ids=torch.tensor(ids, device=args.device),
                                    attention_mask=torch.tensor(masks, device=args.device))
                    # print("encoder output:", tuple(outputs.shape))
                    # assert torch.isfinite(outputs).all().item()
                    vectors.append(torch.nn.functional.normalize(outputs.float(), dim=1).cpu())
                    # print("normalized vector norms:", vectors[-1].norm(dim=1).tolist()[:5])
            return torch.cat(vectors)

        def similarity(a, b):
            return contract_similarity(embed(bytecode_id(a)), embed(bytecode_id(b)))

        details = {"checkpoint": args.checkpoint.name, "encoder": "official_smarttrans",
                   "aggregation": "symmetric_mean_best_function_cosine", "max_length": 512}
        score_file(args.inputs, args.output, "SmartTrans", similarity, details)
    finally:
        sys.path.remove(with_repo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export-bytecodes", help="Export unique .hex files for Gigahorse")
    export.add_argument("inputs", type=Path)
    export.add_argument("--output", type=Path, required=True)
    evaluate = commands.add_parser("score")
    evaluate.add_argument("inputs", type=Path)
    evaluate.add_argument("--repo", type=Path, required=True)
    evaluate.add_argument("--pretrain", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument("--facts", type=Path, help="Gigahorse .temp directory")
    source.add_argument("--functions", type=Path, help="Trusted SmartTrans preprocess_cfg.pkl keyed by exported filenames")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    try:
        if args.command == "export-bytecodes":
            count = export_bytecodes(args.inputs, args.output)
            print(f"Wrote {count} bytecodes to {args.output}")
        else:
            if args.batch_size < 1:
                raise ValueError("Batch size must be positive")
            score(args)
    except (ImportError, OSError, KeyError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
