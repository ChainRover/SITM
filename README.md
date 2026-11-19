# SITM Experiment Code

This project implements the SITM bytecode similarity metric and the statistical analysis for the paper's experiments. Results are saved as JSON and CSV files.

Original project code is **source-available for noncommercial use** under the
[PolyForm Noncommercial License 1.0.0](LICENSE). Copyright (c) 2026
Xiamen University of Technology、Xiamen Municipal Public Technology Service Platform for Software Supply Chain Security. Preserve [NOTICE](NOTICE).
Commercial uses outside the license's permitted purposes require separate
permission from the copyright holders.

## Installation

Use Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
```

Bytecode scoring reads the data files directly. Compiling Solidity source and running smoke tests require solc; py-solc-x downloads any missing compiler versions. For offline use, specify the compiler directory with `--solc-dir`. Compiler filenames include `solc-v0.5.17` and `solc-v0.8.20`.

## Quick Start

```bash
python -u run.py --exp smoke --out smoke-results
```

The smoke test uses a manually written Probe contract. The negative sample uses CH by default; use `--smoke-mutation SH|RS|PT|ET` to select another type. Positive samples come from the original Probe and two source variants that write different constants. The generated source files and inputs are saved in `smoke-results/fixtures`. This command checks the compilation, scoring, and statistical analysis workflow.

If compilers 0.5.0 and 0.8.20 are already available, run:

```bash
python -u run.py --exp smoke --solc-dir .sitm-solc --legacy-version 0.5.0 --out smoke-results
```

Reanalyze the generated examples:

```bash
python -u run.py smoke-results/fixtures/pairs.jsonl --exp all --calibrate --bootstrap 20 --out smoke-tables
```

## Running the Paper's Experiments

Links to the original corpus datasets, Smartbugs and ScrawId, are provided in this project. Download the original contract files from these links or the source files from their official GitHub repositories. The dataset splits are described in the paper.

The remaining dataset files are available at https://my.feishu.cn/drive/folder/Tzw2fb5M0lTI2KdlkYAcdQvSnUe

Place the frozen data files in `data`, or pass their paths on the command line.

```bash
python -u run.py --exp all --dataset-dir data --calibrate --skip-llm --out results
```

Run all available datasets, including the downstream experiments:

```bash
python -u run.py --exp all --dataset-dir data --llm-dir data/llm --calibrate --out results-all
```

If E2V and SmartTrans scores for each pair and differential replay records are available, add the following arguments:

```bash
python -u run.py --exp all --dataset-dir data --calibrate --out results-full \
  --scores data/e2v_scores.jsonl --scores data/smarttrans_scores.jsonl \
  --replays data/replay_observations.jsonl
```

Multiple JSONL files can be passed directly:

```bash
python -u run.py data/pairs.jsonl data/positives.jsonl --exp all --out results
```

The supplied frozen pairs are used by default. Missing Mut-Set positive samples are sampled from Opt-Pair only when `--positive-count` is explicitly set. `--noise-count` generates local examples with noise appended to the bytecode. For an exact comparison with the paper's tables, provide the original pairs, noise artifacts, and compiler configurations.

| `--exp`    | Experiments                                                  |
| ---------- | ------------------------------------------------------------ |
| `all`      | All experiments supported by the current inputs              |
| `core`     | All experiments that do not use LLMs                         |
| `1`        | Robustness to optimizer settings                             |
| `2`        | Stability across compiler versions                           |
| `3`        | Mut-Set classification and temporal holdout tests, where available |
| `4`        | Robustness to noise                                          |
| `5`        | Ablation studies                                             |
| `6`        | Scoring runtime                                              |
| `7`        | Parameter sensitivity on the mixed set                       |
| `8`        | α calibration and search over gating settings and thresholds |
| `9`, `10`  | Field traceability rate                                      |
| `11`, `12` | LLM evaluation                                               |

`--threshold` defaults to 0.60. Equivalent pairs form the positive class; a pair is classified as equivalent when its score is greater than or equal to the threshold. `--bootstrap` defaults to 1,000 iterations for full runs, with positive and negative samples drawn separately with replacement. PR-AUC is computed using trapezoidal integration of the PR curve. The default random seed is 42.

## Input Format

Experiment data paths are selected by `input_paths()` in `run.py` and then read by `read_pairs()` in `data.py`. `--dataset-dir` specifies the directory containing files in the legacy format. If neither a directory nor positional arguments are supplied, the program looks in `dataset/smartbugs/` under the current working directory. It does not automatically locate other data directories on the machine.

Your existing Linux data path can be specified directly:

```bash
python -u run.py --exp all \
  --dataset-dir /home/user/zzb/mac4d/dataset/smartbugs \
  --calibrate --skip-llm --out results-all
```

Other path arguments are:

| Argument                    | Data read                                                    |
| --------------------------- | ------------------------------------------------------------ |
| Positional argument `pairs` | One or more JSONL files containing pairs, specified directly |
| `--positive-file`           | Additional Mut-Set positive samples; can be specified more than once |
| `--llm-dir`, `--llm-file`   | LLM pairs archived by model                                  |
| `--scores`                  | E2V and SmartTrans scores matched by pair ID; can be specified more than once |
| `--replays`                 | Differential replay observations for sampled pairs; can be specified more than once |
| `--out`                     | Output directory for the results summarized in this run      |

When only `--dataset-dir` is supplied, legacy LLM files are located by searching the sibling directory `llm/` for `sitm_downstream_*_pairs.jsonl`. If none are found, the program tries `llm_common/common_addresses_flat.jsonl` at the same level. LLM paths can be specified explicitly for full runs.

Data locations for the other commands are listed below:

| File            | Input location                                               |
| --------------- | ------------------------------------------------------------ |
| `prepare.py`    | Source or pair manifests supplied as positional arguments; source files are located relative to the manifest directory, or under `--source-root` if specified |
| `e2v.py`        | `prepare` reads the contracts manifest, `train` reads the training JSONL, and `score` reads the pair files and the model directory specified by `--model` |
| `smarttrans.py` | Reads pair files supplied as positional arguments; the model is specified by `--repo`, `--pretrain`, and `--checkpoint`, and function data by `--facts` or `--functions` |
| `demo.py`       | Reads the Solidity source embedded in the file               |
| `smoke.py`      | Uses the embedded Probe source; generated data is saved under `fixtures/` in the output directory for this run |

The common pair format is shown below. `a` and `b` are runtime bytecode.

```json
{"id":"mut:contract-001:sh-1","contract_id":"contract-001","dataset":"mut","label":0,"mutation":"SH","a":"600160005500","b":"600160015500","traceability":"static"}
```

| Filename recognized by default | `dataset`     | Size and purpose in the paper                                |
| ------------------------------ | ------------- | ------------------------------------------------------------ |
| `dataset_opt_pair.jsonl`       | `opt`         | 24,079 pairs equivalent across optimizer settings            |
| `dataset_ver_pair.jsonl`       | `ver`         | 520 minor pairs and 158 major pairs                          |
| `dataset_mut_pair.jsonl`       | `mut`         | 1,000 positive and 1,000 negative pairs, with 200 negative pairs for each of the five mutation types |
| `dataset_noise_pair.jsonl`     | `noise`       | 500 pairs for each of the four noise types                   |
| `dataset_calib.jsonl`          | `calibration` | 500 equivalent pairs for α search                            |
| `dataset_mixed.jsonl`          | `mixed`       | 500 positive and 500 negative pairs for parameter selection and sensitivity analysis |
| `dataset_temporal.jsonl`       | `temporal`    | 150 contracts and 450 pairs, including 150 positive pairs across optimizer settings |

The legacy format also accepts the fields `bytecode_original/bytecode_mutated`, `bytecode_unopt/bytecode_opt`, and `bytecode_base/bytecode_target`. For legacy files without addresses, record indices are used, and these records are counted in the summary. If a legacy `dataset_calib.jsonl` contains negative labels, the program prints a message and reads those records as mixed data. Records in the common format explicitly labeled calibration should contain only equivalent samples.

If only negative samples in the five-field format are available, supply equivalent positive samples separately:

```bash
python -u run.py --exp 3 --dataset-dir data --positive-file data/mut_positives.jsonl --out results-mut
```

Use `group=minor|major` to specify the group for `ver` records. For static/dynamic field analysis of `CH` and `SH`, use `traceability=static|dynamic`.

In `noise` records, `a` is the reference bytecode, `b` is the noisy bytecode, and `clean_b` is the corresponding version before noise was added. Set `noise` to `metadata`, `junk`, `deadcode`, or `combined`.



## Compilation and Differential Replay

`prepare.py` provides the subcommands `prepare`, `make-recipes`, `prepare-mutants`, `prepare-llm`, `split`, `export-model-inputs`, and `replay`. Source manifests contain `id`, `target`, `sources`, and `versions`, along with optional compiler settings.

```bash
python prepare.py prepare data/contracts.jsonl --output pairs.jsonl
python prepare.py make-recipes data/contracts.jsonl --output recipes.jsonl
python prepare.py prepare-mutants recipes.jsonl --source-root data --output mutants.jsonl
python prepare.py replay data/pairs.jsonl --cases data/replay_cases.jsonl --output replay_observations.jsonl
```

Replay requires Anvil. Each case uses the pair ID and supplies the same initial account state for both sides, along with the sender, calldata, value, and block context for 5 transactions. Archived observations contain calls, writes, logs, exit, storage_root, and receipts_root. Table 5 reports confirmation rates using both all sampled cases and executable cases as denominators.

### Python dependencies and external tools

Python packages are installed separately using `requirements.txt`,
`requirements-models.txt`, and `requirements-dev.txt`. Their licenses remain in
the packages' distributions. This repository does not bundle these packages.
Keep their license and notice files when redistributing an environment or image.

Relevant upstream projects include:

| Material | Upstream source | Use in SITM |
| --- | --- | --- |
| NumPy | https://numpy.org/ | Numerical statistics |
| SciPy | https://scipy.org/ | Statistical tests |
| scikit-learn | https://scikit-learn.org/ | Classification metrics |
| NetworkX | https://networkx.org/ | CFG comparison |
| RapidFuzz | https://github.com/rapidfuzz/RapidFuzz | Edit distance |
| py-solc-x | https://github.com/iamdefinitelyahuman/py-solc-x | Compiler integration |
| pyevmasm | https://github.com/crytic/pyevmasm | Opcode descriptions |
| cbor2 | https://github.com/agronholm/cbor2 | Metadata parsing |
| PyCryptodome | https://www.pycryptodome.org/ | Keccak hashing |
| semantic-version | https://github.com/rbarrois/python-semanticversion | Compiler constraints |
| Solidity / solc | https://github.com/ethereum/solidity | External compiler |
| Foundry / Anvil | https://github.com/foundry-rs/foundry | External replay environment |
| PyTorch | https://pytorch.org/ | Optional model training and inference |
| Hugging Face Transformers / Tokenizers | https://github.com/huggingface/transformers | Optional model implementation |

This is a source inventory, not a certification of every installed package or
transitive dependency. Record exact versions and preserve upstream license files
for the environment actually distributed.

### Publication inventory

The public source repository currently excludes `data/`, compiler binaries,
model weights, and JSONL/CSV artifacts through `.gitignore`. If these are later
distributed in a supplement or release archive, include their provenance,
permissions, and required notices with that archive. Do not infer permission
solely from public availability on a block explorer or file-sharing service.
