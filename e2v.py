import argparse
import json
from functools import lru_cache
from pathlib import Path
from random import Random

from evm_sit import bytecode_bytes, disassemble, strip_metadata
from model_scores import artifact_hashes, bytecode_id, file_sha256, score_file
from data import build, read_jsonl, standard_input, write_jsonl


MAX_LENGTH = 512
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
REPRESENTATIONS = ("opcode-only", "operand-bytes")
BYTE_TOKENS = [f"IMM_{value:02x}" for value in range(256)]


def opcode_text(code, representation="opcode-only", *, strip_meta=True):
    """Encode the published opcode baseline or an explicitly separate variant.

    Opcode-only intentionally discards PUSH values, so it cannot distinguish
    same-width immediate-only PT mutations. Operand bytes require retraining.
    """
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown E2V representation: {representation}")
    tokens = []
    normalized = strip_metadata(code) if strip_meta else bytecode_bytes(code)
    for ins in disassemble(normalized):
        tokens.append(ins.name)
        if representation == "operand-bytes" and ins.name.startswith("PUSH") and ins.name != "PUSH0":
            width = int(ins.name[4:])
            tokens.extend(f"IMM_{value:02x}" for value in ins.immediate.to_bytes(width, "big"))
        if len(tokens) >= MAX_LENGTH - 2:
            break
    return " ".join(tokens[:MAX_LENGTH - 2])


def model_representation(directory, requested=None):
    path = Path(directory) / "representation.json"
    if path.is_file():
        archived = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(archived, dict):
            raise ValueError(f"Expected an object in E2V representation manifest: {path}")
        representation = archived.get("representation")
        if representation not in REPRESENTATIONS or archived.get("max_length") != MAX_LENGTH:
            raise ValueError(f"Unsupported E2V representation manifest: {path}")
        if archived.get("preprocessing") != "solidity_cbor_metadata_stripped":
            raise ValueError(f"Unsupported E2V preprocessing manifest: {path}")
        status = "archived"
    else:
        # Legacy checkpoints had no representation manifest. They can only be
        # interpreted as the old opcode-only baseline, never the new variant.
        representation, status = "opcode-only", "legacy_assumed_opcode_only"
    if requested is not None and requested != representation:
        raise ValueError(f"Checkpoint uses {representation}, requested {requested}; retrain the requested representation")
    return representation, status


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

    representation = getattr(args, "representation", "opcode-only")
    texts = []
    seen = set()
    tokens = set()
    for row in read_jsonl(args.train_data):
        identifier = row["contract_id"].lower()
        if row.get("split") != "train" or identifier in seen:
            raise ValueError(f"Expected a unique train contract: {identifier}")
        seen.add(identifier)
        text = opcode_text(row["bytecode"], representation)
        # if len(texts) < 3:
        #     print("training contract:", identifier, "opcode tokens:", len(text.split()))
        if text:
            texts.append(text)
            tokens.update(text.split())
    if not texts:
        raise ValueError("No opcodes in the training corpus")
    if representation == "operand-bytes":
        # A fixed byte alphabet retains unseen immediates instead of mapping
        # every unseen PUSH value to [UNK]. This is a new trained baseline.
        tokens.update(BYTE_TOKENS)
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
    model.config.sitm_representation = representation
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
    manifest = {"schema_version": 1, "representation": representation, "max_length": MAX_LENGTH,
                "preprocessing": "solidity_cbor_metadata_stripped"}
    (args.output / "representation.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    details = {"training": "local_mlm", "seed": args.seed, "epochs": args.epochs,
               "batch_size": args.batch_size, "learning_rate": args.learning_rate,
               "training_contracts": len(texts), "optimizer_steps": updates,
               "representation": representation, "train_manifest_sha256": file_sha256(args.train_data),
               "training_code_sha256": file_sha256(Path(__file__)),
               "training_contract_ids_sha256": bytecode_id("\n".join(sorted(seen)).encode()),
               "training_text_sha256": bytecode_id("\n".join(texts).encode())}
    (args.output / "training.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")


def score_model(args):
    representation, representation_status = model_representation(args.model, getattr(args, "representation", None))
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
    configured = getattr(model.config, "sitm_representation", "opcode-only")
    if configured != representation:
        raise ValueError("Saved model configuration and representation manifest disagree; retrain the requested representation")
    if representation == "operand-bytes" and not set(BYTE_TOKENS) <= tokenizer.get_vocab().keys():
        raise ValueError("Operand-aware checkpoint must contain all 256 immediate-byte tokens")
    model.to(args.device).eval()

    @lru_cache(maxsize=512)
    def embed(code):
        # score_file already applied the single shared preprocessing pass.
        inputs = tokenizer(opcode_text(code, representation, strip_meta=False), return_tensors="pt",
                           truncation=True, max_length=MAX_LENGTH)
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
        # Roundoff can otherwise report 1.0000001 for identical vectors.
        return float(cosine_similarity(embed(a), embed(b), dim=0).clamp(-1.0, 1.0))

    details = {"checkpoint": args.model.name, "pooling": args.pooling, "max_length": MAX_LENGTH,
               "checkpoint_files_sha256": artifact_hashes(args.model),
               "representation": representation, "representation_status": representation_status,
               "scoring_code_sha256": file_sha256(Path(__file__)),
               "training_provenance": {"status": "unprovided"}}
    training = args.model / "training.json"
    if training.is_file():
        archived = json.loads(training.read_text(encoding="utf-8"))
        if not isinstance(archived, dict):
            raise ValueError("Expected an object in E2V training metadata")
        if archived.get("representation", "opcode-only") != representation:
            raise ValueError("Training metadata and checkpoint representation disagree")
        details["training_provenance"] = {"status": "archived_metadata", "file_sha256": file_sha256(training),
                                          "metadata": archived}
    method = "E2V" if representation == "opcode-only" else "E2V_operand"
    diagnostics = lambda a, b: {"identical_model_inputs": opcode_text(a, representation, strip_meta=False)
                               == opcode_text(b, representation, strip_meta=False)}
    score_file(args.inputs, args.output, method, similarity, details, diagnostics)


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
    train.add_argument("--representation", choices=REPRESENTATIONS, default="opcode-only",
                       help="operand-bytes is a separate baseline requiring a new checkpoint")
    score = commands.add_parser("score")
    score.add_argument("inputs", type=Path)
    score.add_argument("--model", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--pooling", choices=("mean", "cls"), default="mean")
    score.add_argument("--device", default="cpu")
    score.add_argument("--representation", choices=REPRESENTATIONS,
                       help="Must match the saved checkpoint; inferred when omitted")
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
