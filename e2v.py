import argparse
import json
from functools import lru_cache
from pathlib import Path
from random import Random

from evm_sit import bytecode_bytes, disassemble, strip_metadata
from model_scores import score_file
from data import build, read_jsonl, standard_input, write_jsonl


MAX_LENGTH = 512
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def opcode_text(code):
    return " ".join(ins.name for ins in disassemble(bytecode_bytes(code))[:MAX_LENGTH - 2])


def prepare_data(args):
    records = []
    root = args.source_root or args.contracts.parent
    for spec in read_jsonl(args.contracts):
        if spec.get("split") != "train":
            continue
        version = args.version or spec.get("opt_version")
        if version is None:
            raise ValueError(f"Set opt_version or --version for {spec['id']}")
        config = standard_input(spec, root, version, False)
        runtime = build(config, spec["target"], version, args.solc_dir).bytecode
        records.append({"contract_id": spec["id"], "split": "train", "bytecode": runtime})
    if not records:
        raise ValueError("No contracts are assigned to the train split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} training contracts to {args.output}")


def train_model(args):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from tokenizers.processors import TemplateProcessing
    from transformers import DataCollatorForLanguageModeling, DebertaV2Config
    from transformers import DebertaV2ForMaskedLM, PreTrainedTokenizerFast

    texts = []
    seen = set()
    tokens = set()
    for row in read_jsonl(args.train_data):
        identifier = row["contract_id"].lower()
        if row.get("split") != "train" or identifier in seen:
            raise ValueError(f"Expected a unique train contract: {identifier}")
        seen.add(identifier)
        text = opcode_text(strip_metadata(row["bytecode"]))
        # if len(texts) < 3:
        #     print("training contract:", identifier, "opcode tokens:", len(text.split()))
        if text:
            texts.append(text)
            tokens.update(text.split())
    if not texts:
        raise ValueError("No opcodes in the training corpus")
    vocabulary = SPECIAL + sorted(tokens)
    # print("training contracts:", len(texts), "unique contracts:", len(seen))
    # print("vocabulary:", len(vocabulary), "special tokens:", SPECIAL)
    vocab = {token: index for index, token in enumerate(vocabulary)}
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    backend.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, model_max_length=MAX_LENGTH, pad_token="[PAD]",
        unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
    )
    torch.manual_seed(args.seed)
    model = DebertaV2ForMaskedLM(DebertaV2Config(
        vocab_size=len(vocab), hidden_size=768, num_hidden_layers=4, num_attention_heads=12,
        intermediate_size=768, max_position_embeddings=MAX_LENGTH, pad_token_id=vocab["[PAD]"],
        relative_attention=True, pos_att_type=["p2c", "c2p"], share_att_key=True,
    )).to(args.device)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm_probability=0.15)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    rng = Random(args.seed)
    model.train()
    updates = 0
    for epoch in range(args.epochs):
        rng.shuffle(texts)
        total_loss, steps = 0.0, 0
        for start in range(0, len(texts), args.batch_size):
            batch = tokenizer(texts[start:start + args.batch_size], truncation=True,
                              max_length=MAX_LENGTH, return_special_tokens_mask=True)
            examples = [{key: batch[key][i] for key in batch} for i in range(len(batch["input_ids"]))]
            masked = collator(examples)
            # print("training batch:", epoch, start, masked["input_ids"].shape)
            # print("masked targets:", int((masked["labels"] != -100).sum()))
            if not (masked["labels"] != -100).any():
                continue
            masked = {key: value.to(args.device) for key, value in masked.items()}
            optimizer.zero_grad()
            loss = model(**masked).loss
            # assert torch.isfinite(loss).item()
            # print("batch loss:", float(loss.detach().cpu()))
            loss.backward()
            optimizer.step()
            steps += 1
            total_loss += loss.item()
        updates += steps
        if steps:
            print(f"Epoch {epoch + 1}: loss={total_loss / steps:.4f}")
    if not updates:
        raise ValueError("No masked tokens were sampled. Use more training bytecode or more epochs.")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    details = {"training": "local_mlm", "seed": args.seed, "epochs": args.epochs,
               "batch_size": args.batch_size, "learning_rate": args.learning_rate,
               "training_contracts": len(texts), "optimizer_steps": updates}
    (args.output / "training.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")


def score_model(args):
    import torch
    from torch.nn.functional import cosine_similarity
    from transformers import AutoModel, AutoTokenizer

    if not (args.model / "config.json").is_file():
        raise ValueError(f"No saved model configuration in {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model, loading = AutoModel.from_pretrained(args.model, local_files_only=True, output_loading_info=True)
    # print("encoder loading:", loading)
    # print("encoder type / hidden size:", model.config.model_type, model.config.hidden_size)
    if loading["missing_keys"] or model.config.model_type != "deberta-v2" or model.config.hidden_size != 768:
        raise ValueError("Expected a complete DeBERTa-v2 encoder with 768-dimensional vectors")
    model.to(args.device).eval()

    @lru_cache(maxsize=512)
    def embed(code):
        inputs = tokenizer(opcode_text(code), return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
        inputs = inputs.to(args.device)
        # print("encoded tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())[:30])
        # print("attention tokens:", int(inputs["attention_mask"].sum()), "pooling:", args.pooling)
        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state[0]
            vector = hidden[0] if args.pooling == "cls" else hidden[inputs["attention_mask"][0].bool()].mean(0)
            # assert vector.shape == (768,)
            # assert torch.isfinite(vector).all().item()
            # print("embedding norm:", float(vector.norm().cpu()))
            # print("输入", inputs["input_ids"].shape, "隐藏层", hidden.shape)
            # print("池化向量", vector.shape)
            # 去掉首尾特殊 token 的池化对照，只跑一小批样本观察差值。
            # valid = inputs["attention_mask"][0].bool().clone()
            # positions = valid.nonzero(as_tuple=False).flatten()
            # if len(positions) > 2:
            #     valid[positions[0]] = False
            #     valid[positions[-1]] = False
            #     content_vector = hidden[valid].mean(0)
            #     print("去特殊 token 后的差值", (vector - content_vector).norm())
            return vector.cpu()

    def similarity(a, b):
        return float(cosine_similarity(embed(a), embed(b), dim=0))

    details = {"checkpoint": args.model.name, "pooling": args.pooling, "max_length": MAX_LENGTH}
    training = args.model / "training.json"
    if training.is_file():
        details.update(json.loads(training.read_text(encoding="utf-8")))
    score_file(args.inputs, args.output, "E2V", similarity, details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("contracts", type=Path)
    prepare.add_argument("--version")
    prepare.add_argument("--source-root", type=Path)
    prepare.add_argument("--solc-dir", type=Path)
    prepare.add_argument("--output", type=Path, required=True)
    train = commands.add_parser("train")
    train.add_argument("train_data", type=Path)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--learning-rate", type=float, default=5e-5)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cpu")
    score = commands.add_parser("score")
    score.add_argument("inputs", type=Path)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    score.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            prepare_data(args)
        elif args.command == "train":
            if min(args.epochs, args.batch_size) < 1 or args.learning_rate <= 0:
                raise ValueError("Epochs, batch size and learning rate must be positive")
            train_model(args)
        else:
            score_model(args)
    except (ImportError, OSError, KeyError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
