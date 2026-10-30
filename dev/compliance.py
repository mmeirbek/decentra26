"""The TZ audit we run every third cycle: the source rules and the margins of a Kaggle T4 run.

    python3 dev/compliance.py runs_kaggle/out_c3

Source side (the jury reads solution/): banned kernel libraries, torch.compile and CUDA graphs,
random number generators, environment and file reads, the example's kernels and lookalikes, and
harness/ tests/ task/ templates/ untouched since the starter kit. Run side: every check of
check_rules.py, with how far each number is from its limit. Deliverables: README sections and
SUBMISSION.md.
"""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STARTER = "d81ac66"
sys.path.insert(0, str(REPO / "harness"))


def line(ok, what, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'}  {what}" + (f"  ({detail})" if detail else ""))
    return ok


def source_checks():
    from check_rules import source_kernels  # the jury's own detector
    from common import BANNED_MODULES

    ok = True
    files = [p for p in sorted((REPO / "solution").glob("*.py")) if not p.name.startswith("_")]
    code = {p.name: p.read_text() for p in files}
    prod = {n: c for n, c in code.items() if not n.startswith("test_")}
    imports = set()
    for name, text in code.items():
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Import):
                imports |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imports.add(node.module.split(".")[0])
    allowed = {"torch", "triton", "transformers", "peft", "trl", "bitsandbytes", "soup_cli",
               "threading", "time", "os", "pytest", "types", "solution"}
    ok &= line(not (imports & set(BANNED_MODULES)), "no banned kernel library imported",
               ", ".join(sorted(imports & set(BANNED_MODULES))) or "none")
    ok &= line(imports <= allowed, "only allowed packages imported", ", ".join(sorted(imports)))
    patterns = {
        "torch.compile / CUDA graphs": r"torch\.compile|CUDAGraph|cuda\.graph|make_graphed",
        "random number generators (besides torch.native_dropout)":
            r"torch\.(rand|randn|randint|randperm|manual_seed|bernoulli|normal)|\brandom\.|np\.random",
        "environment variables": r"os\.environ|getenv",
        "file reads": r"\bopen\(|read_text|read_bytes|from_pretrained|np\.load|torch\.load",
        "allocator / cache games": r"empty_cache|memory_fraction|ALLOC_CONF|set_allocator",
    }
    def code_only(text):
        """The source with strings and comments blanked out (docstrings mention from_pretrained)."""
        import io
        import tokenize

        out = []
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type not in (tokenize.STRING, tokenize.COMMENT):
                out.append((tok.start[0], tok.string))
        lines = {}
        for row, string in out:
            lines.setdefault(row, []).append(string)
        return {row: " ".join(parts) for row, parts in lines.items()}

    stripped = {n: code_only(c) for n, c in prod.items()}
    for what, pattern in patterns.items():
        hits = [f"{n}:{row}" for n, rows in stripped.items() for row, l in rows.items()
                if re.search(pattern, l)]
        ok &= line(not hits, f"no {what} in solution code", ", ".join(hits) or "none")
    drop = [n for n, c in prod.items() if "native_dropout" in c]
    line(True, "the only RNG use is torch.native_dropout (the base's own LoRA dropout draws)",
         ", ".join(drop))
    own, copies, lookalikes, banned, broken = source_kernels(REPO / "solution")
    ok &= line(bool(own) and not copies and not lookalikes and not banned and not broken,
               f"{len(own)} own @triton.jit kernels, no example copies or lookalikes",
               ", ".join(sorted(own)))
    changed = subprocess.run(["git", "-C", str(REPO), "diff", "--stat", STARTER, "--",
                              "harness", "tests", "task", "templates"],
                             capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--short", "--", "harness", "tests",
                            "task", "templates"], capture_output=True, text=True).stdout.strip()
    ok &= line(not changed and not dirty, "harness/ tests/ task/ templates/ as in the starter kit",
               changed or dirty or "unchanged")
    return ok


def run_checks(folder):
    base = Path(folder) / "out" if (Path(folder) / "out").exists() else Path(folder)
    checks = json.loads((base / "checks.json").read_text())["checks"]
    result = json.loads((base / "result.json").read_text())
    ok = all(c["ok"] for c in checks)
    for c in checks:
        line(c["ok"], c["check"], str(c["detail"])[:150])
    ratios = result["summary"]["ratios"]
    sol = result["runs"]["solution"]
    share = min(sum(r["metrics"]["own_kernel_ms"].values()) / 1000 / r["metrics"]["timed_loop_s"]
                for r in sol)
    line(True, "margins", f"kernel share {100 * share:.1f}% (limit 5%), eval loss "
         f"{ratios.get('quality_delta_pct')}% (limit +0.5%), time x{1 / ratios['speedup']:.3f} "
         f"and memory x{1 / ratios['memory_ratio']:.3f} of the base (limit x1.10)")
    print(f"      s = {ratios['speedup']:.3f}, m = {ratios['memory_ratio']:.3f}")
    return ok


def deliverables():
    readme = (REPO / "README.md").read_text()
    sections = ["## Что сделано", "## Как запустить", "## Результаты", "Источники"]
    ok = line(all(s in readme for s in sections), "README has what / how to run / results / sources")
    placeholders = re.findall(r"RESULT_\w+|KAGGLE_LINK|COMMIT_LINK|FINAL_COMMIT|RESULTS_DETAILS",
                              readme)
    ok &= line(not placeholders, "README filled in", ", ".join(sorted(set(placeholders))) or "yes")
    sub = REPO / "SUBMISSION.md"
    ok &= line(sub.exists(), "SUBMISSION.md in the repository root")
    return ok


if __name__ == "__main__":
    print("== source")
    s = source_checks()
    r = True
    if len(sys.argv) > 1:
        print(f"== Kaggle run {sys.argv[1]}")
        r = run_checks(sys.argv[1])
    print("== deliverables")
    d = deliverables()
    print("\nALL GOOD" if s and r and d else "\nNOT YET")
