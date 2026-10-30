"""Build a private Kaggle notebook that runs the jury's flow on our solution/, and push it to a T4.

    python3 dev/kaggle_push.py --user <kaggle username> [--repeats 2] [--profile]
    uvx --from kaggle==2.2.5 kaggle kernels push -p runs_kaggle/build --accelerator NvidiaTeslaT4

The notebook does what the jury does: it clones the public starter kit at the commit our repository
was made from, puts in only our solution/ (embedded in the notebook, so no GitHub token is needed),
installs the pinned packages, and runs prepare_data, `pytest solution`, run_task.py, check_rules.py
and score.py. A last cell copies results.json, checks.json and each run's metrics and log to
/kaggle/working/out and prints a summary; `kaggle kernels output` downloads them. For development
only: the submitted notebook is templates/kaggle_notebook.ipynb with REPO_URL and COMMIT filled in.
"""

import argparse
import base64
import io
import json
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KIT = "https://github.com/MakazhanAlpamys/soup-kernel-challenge.git"
KIT_COMMIT = "c7c2fcd"


NOTEBOOK_METADATA = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"},
                     "language_info": {"name": "python"}}


def solution_archive(replace=(), extra=()):
    """solution/ as a base64 tar.gz; `replace` holds (file, old, new) source edits and `extra`
    (archive name, source path) files added for experiments."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, source in extra:
            tar.add(source, arcname=name)
        for path in sorted((REPO / "solution").rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            name = str(path.relative_to(REPO))
            data = path.read_bytes()
            for file, old, new in replace:
                if name == file:
                    if old.encode() not in data:
                        raise SystemExit(f"{old!r} not found in {file}")
                    data = data.replace(old.encode(), new.encode())
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode()


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [line + "\n" for line in lines]}


def notebook(archive, repeats, profile, only_solution=False, tune=False, tests_only=False):
    template = json.loads((REPO / "templates" / "kaggle_notebook.ipynb").read_text())
    install = [c for c in template["cells"] if c["cell_type"] == "code"][1]  # GPU check + pip
    run_args = f"--repeats {repeats}" + (" --profile" if profile else "") + \
        (" --only solution" if only_solution else "")
    clone = code("import base64, io, shutil, subprocess, tarfile",
                 "shutil.rmtree('/kaggle/working/repo', ignore_errors=True)",
                 f"subprocess.run(['git', 'clone', '-q', '{KIT}', '/kaggle/working/repo'], check=True)",
                 f"subprocess.run(['git', '-C', '/kaggle/working/repo', 'checkout', '-q', '{KIT_COMMIT}'], check=True)",
                 "shutil.rmtree('/kaggle/working/repo/solution')",
                 f"ARCHIVE = '{archive}'",
                 "tarfile.open(fileobj=io.BytesIO(base64.b64decode(ARCHIVE))).extractall('/kaggle/working/repo')")
    if tests_only:
        return {"cells": [install, clone, code(
            "!cd /kaggle/working/repo && python -m pytest solution -x -q -p no:cacheprovider "
            "--tb=long -W ignore 2>&1 | tail -150")], "metadata": NOTEBOOK_METADATA,
            "nbformat": 4, "nbformat_minor": 5}
    cells = [
        install,
        clone,
        code("import torch, torch.nn.functional as F",
             "from torch.nn.attention import SDPBackend, sdpa_kernel",
             "q = torch.randn(1, 12, 300, 128, device='cuda', dtype=torch.float16, requires_grad=True)",
             "k = torch.randn(1, 2, 300, 128, device='cuda', dtype=torch.float16, requires_grad=True)",
             "v = torch.randn(1, 2, 300, 128, device='cuda', dtype=torch.float16, requires_grad=True)",
             "try:",
             "    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):",
             "        F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True).sum().backward()",
             "    print('PROBE efficient attention with enable_gqa: supported')",
             "except Exception as exc:",
             "    print('PROBE efficient attention with enable_gqa: NOT supported', repr(exc)[:200])"),
        code("!cd /kaggle/working/repo && python task/prepare_data.py",
             "!cd /kaggle/working/repo && python -m pytest solution -q 2>&1 | tail -15"),
        code("!cd /kaggle/working/repo && python -m solution._tune 2>&1 | tail -150" if tune
             else "print('no tuning in this run')"),
        code(f"!cd /kaggle/working/repo && python harness/run_task.py {run_args}"),
        code("!cd /kaggle/working/repo && python harness/check_rules.py",
             "!cd /kaggle/working/repo && python harness/score.py runs/result.json"),
        code("import json, pathlib, shutil, collections",
             "R = pathlib.Path('/kaggle/working/repo/runs'); OUT = pathlib.Path('/kaggle/working/out')",
             "OUT.mkdir(exist_ok=True)",
             "try:",
             "    for name in ('result.json', 'checks.json'):",
             "        if (R / name).exists(): shutil.copy(R / name, OUT / name)",
             "    for run in R.iterdir():",
             "        if run.is_dir():",
             "            for name in ('metrics.json', 'train.log', 'profile.json'):",
             "                if (run / name).exists(): shutil.copy(run / name, OUT / f'{run.name}_{name}')",
             "            cache = run / 'triton_cache'",
             "            if cache.exists():",
             "                (OUT / f'{run.name}_compiled.json').write_text(json.dumps(collections.Counter(p.stem for p in cache.rglob('*.cubin'))))",
             "    for log in R.glob('eval_*.log'): shutil.copy(log, OUT / log.name)",
             "    res = json.loads((R / 'result.json').read_text())",
             "    print(json.dumps(res['summary'], indent=1))",
             "except Exception as exc:",
             "    print('summary failed:', repr(exc))",
             "shutil.rmtree('/kaggle/working/repo', ignore_errors=True)"),
    ]
    return {"cells": cells, "metadata": NOTEBOOK_METADATA, "nbformat": 4, "nbformat_minor": 5}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--user", required=True)
    parser.add_argument("--slug", default="soup-kernel-dev")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--no-recompute-mlp", action="store_true",
                        help="experiment: run with model.RECOMPUTE_MLP = False")
    parser.add_argument("--tests-only", action="store_true",
                        help="only run `pytest solution` on the GPU, with full tracebacks")
    parser.add_argument("--only-solution", action="store_true",
                        help="train only the solution (quicker; the checks that need the base fail)")
    parser.add_argument("--tune", action="store_true",
                        help="experiment: time kernel block sizes on the T4 (dev/tune_probe.py)")
    parser.add_argument("--probe-first-step", action="store_true",
                        help="experiment: time the first training steps (dev/first_step_probe.py)")
    parser.add_argument("--out", default=str(REPO / "runs_kaggle" / "build"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    replace = []
    if args.no_recompute_mlp:
        replace.append(("solution/model.py", "RECOMPUTE_MLP = True", "RECOMPUTE_MLP = False"))
    extra = []
    if args.probe_first_step:
        extra.append(("solution/_probe.py", REPO / "dev" / "first_step_probe.py"))
        replace.append(("solution/__init__.py", "    patch_soup_wrapper()\n",
                        "    patch_soup_wrapper()\n    from . import _probe\n    _probe.install()\n"))
    if args.tune:
        extra.append(("solution/_tune.py", REPO / "dev" / "tune_probe.py"))
    (out / "notebook.ipynb").write_text(json.dumps(notebook(solution_archive(replace, extra),
                                                            args.repeats, args.profile,
                                                            args.only_solution, args.tune,
                                                            args.tests_only),
                                                   indent=1))
    meta = {"id": f"{args.user}/{args.slug}", "title": args.slug, "code_file": "notebook.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": True,
            "enable_gpu": True, "enable_internet": True, "dataset_sources": [],
            "competition_sources": [], "kernel_sources": [], "model_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(f"built {out}: push with --accelerator NvidiaTeslaT4")


if __name__ == "__main__":
    main()
