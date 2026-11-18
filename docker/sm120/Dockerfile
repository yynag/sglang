# syntax=docker/dockerfile:1
# Use entrypoint.sh: it resolves the official tag to a digest before building.
ARG CUDA_BASE
FROM ${CUDA_BASE} AS base

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive \
    CUDA_HOME=/usr/local/cuda \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    PYTHONNOUSERSITE=1 \
    PYTHONSAFEPATH=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="" \
    HOME=/work/home \
    XDG_CACHE_HOME=/work/cache \
    TMPDIR=/work/tmp \
    PIP_CACHE_DIR=/work/cache/pip \
    HF_HOME=/work/cache/huggingface \
    TORCH_HOME=/work/cache/torch \
    TORCH_EXTENSIONS_DIR=/work/cache/torch-extensions \
    TRITON_CACHE_DIR=/work/cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/work/cache/inductor \
    CUDA_CACHE_PATH=/work/cache/cuda \
    SGLANG_CACHE_DIR=/work/cache/sglang \
    SGLANG_JIT_CACHE_DIR=/work/cache/sglang/jit

# Keep nvcc/ptxas, C/C++ compilers and Python headers for CUDA JIT in runtime.
# Rust and the explicitly installed Python build backends stay in build only.
RUN . /etc/os-release && test "$VERSION_ID" = 24.04 \
    && nvcc --version | grep -E 'release 13\.' \
    && mkdir -p /work/{home,cache,tmp,log,build} \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates git python3.12 python3.12-dev python3.12-venv \
       gcc g++ make libnuma1 libgomp1 libsndfile1 ffmpeg libibverbs1 \
       ibverbs-providers librdmacm1 \
    && python3.12 -m venv /opt/venv \
    && apt-get clean

FROM base AS build
ARG CUDA_BASE
ARG SOURCE_COMMIT
ARG SOURCE_VERSION
ARG UPSTREAM_BASE
ENV SGLANG_BUILD_RUST_EXTS=all \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=${SOURCE_VERSION} \
    CARGO_HOME=/work/cache/cargo \
    RUSTUP_HOME=/opt/rustup \
    CARGO_TARGET_DIR=/work/build/cargo-target \
    PATH=/work/cache/cargo/bin:${PATH} \
    GIT_OPTIONAL_LOCKS=0 \
    GIT_NO_LAZY_FETCH=1

# The outer context contains an unmodified, clean local Git clone in source/.
# Its nested .dockerignore does not strip tracked build inputs or Git metadata.
COPY source/ /work/source/
WORKDIR /work/source
RUN python3.12 - <<'PY'
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()

commit = git("rev-parse", "HEAD")
assert re.fullmatch(r"[0-9a-f]{40}", commit)
assert commit == os.environ["SOURCE_COMMIT"]
assert git("symbolic-ref", "--short", "HEAD") == "sm120"
assert not git("status", "--porcelain=v1", "--untracked-files=all")
timestamp = git("show", "-s", "--format=%ct", "HEAD")
version = f"0.dev{timestamp}+sm120.g{commit}"
assert version == os.environ["SOURCE_VERSION"]
project = tomllib.loads(Path("python/pyproject.toml").read_text())
assert project["tool"]["setuptools_scm"]["root"] == ".."
out = Path("/out/provenance")
out.mkdir(parents=True)
(out / "source.json").write_text(json.dumps({
    "commit": commit,
    "tree": git("rev-parse", "HEAD^{tree}"),
    "branch": "sm120",
    "shallow": git("rev-parse", "--is-shallow-repository"),
    "version": version,
    "cuda_base": os.environ["CUDA_BASE"],
    "upstream_base_reference_only": os.environ["UPSTREAM_BASE"],
}, indent=2) + "\n")
(out / "build-requirements.txt").write_text(
    "\n".join(project["build-system"]["requires"]) + "\n"
)
default_deps = project["project"]["dependencies"]
flashinfer_match = next(
    (re.search(r"flashinfer_python(?:\[[^\]]*\])?==([0-9A-Za-z.\-]+)", dep)
     for dep in default_deps if "flashinfer_python" in dep),
    None,
)
assert flashinfer_match, "flashinfer_python pin not found in project dependencies"
all_deps = list(default_deps) + [f"flashinfer-cubin=={flashinfer_match.group(1)}"]
(out / "default-requirements.txt").write_text(
    "\n".join(all_deps) + "\n"
)
# flashinfer-cubin is an explicit companion requirement from the FlashInfer index.
# Pass it alongside the complete default dependency file in every resolution.
(out / "constraints.txt").write_text("torch==2.14.1+cu130\n")
PY

RUN --mount=type=cache,target=/work/cache/pip \
    apt-get update \
    && apt-get install -y --no-install-recommends curl pkg-config libssl-dev cmake \
    && python3.12 -m pip install 'pip>=25.3' \
    && python3.12 -m pip install \
       --extra-index-url https://download.pytorch.org/whl/cu130 \
       --constraint /out/provenance/constraints.txt \
       --requirement /out/provenance/build-requirements.txt \
       build \
       --report /out/provenance/build-install.json \
    && python3.12 -m pip install --dry-run --ignore-installed \
       --extra-index-url https://download.pytorch.org/whl/cu130 \
       --extra-index-url https://flashinfer.ai/whl \
       --constraint /out/provenance/constraints.txt \
       --build-constraint /out/provenance/constraints.txt \
       --requirement /out/provenance/default-requirements.txt \
       --report /out/provenance/default-plan.json \
    && curl --proto '=https' --tlsv1.2 --fail --show-error --silent \
       --retry 3 --connect-timeout 30 --max-time 300 \
       https://sh.rustup.rs -o /work/tmp/rustup.sh \
    && bash /work/tmp/rustup.sh -y --no-modify-path --profile minimal --default-toolchain none \
    && python3.12 - <<'PY'
import subprocess
import tomllib
from pathlib import Path

toolchain = tomllib.loads(Path("rust/rust-toolchain.toml").read_text())["toolchain"]
args = ["rustup", "toolchain", "install", toolchain["channel"],
        "--profile", toolchain.get("profile", "minimal")]
for component in toolchain.get("components", []):
    args.extend(["--component", component])
subprocess.run(args, check=True)
subprocess.run(["rustup", "default", toolchain["channel"]], check=True)
PY

RUN --mount=type=cache,target=/work/cache/pip \
    --mount=type=cache,target=/work/cache/cargo/registry \
    --mount=type=cache,target=/work/cache/cargo/git <<'BUILD'
python3.12 - <<'PY'
import json
import runpy
from pathlib import Path
from unittest.mock import patch

import torch

assert str(torch.__version__) == "2.14.1+cu130"
assert torch.version.cuda == "13.0"
with patch("setuptools.setup") as setup:
    state = runpy.run_path("python/setup.py")
extensions = state["_selected_rust_extensions"](
    setup.call_args.kwargs["rust_extensions"]
)
assert extensions
records = []
for extension in extensions:
    metadata = extension._sglang_metadata
    record = {"module": extension.name, "metadata": metadata}
    if header := metadata.get("torch-compat-header"):
        configuration = state["_torch_build_configuration"](
            compat_header=Path(extension._sglang_manifest_path).parent / header,
            python_module=extension.name,
            include_absolute_rpath=False,
        )
        record["torch_build"] = configuration.fingerprint
    records.append(record)
assert any("torch_build" in record for record in records), "TreeCore not selected"
Path("/out/provenance/extensions.json").write_text(json.dumps(records, indent=2) + "\n")
PY
# Build only this project's wheel without isolation: TreeCore uses this Torch.
python3.12 -m build --wheel --no-isolation --outdir /out/wheels ./python
# Resolve the complete default graph from its metadata; dependency source builds
# retain PEP 517 isolation, with the same Torch constraint where needed.
python3.12 -m pip wheel \
    --constraint /out/provenance/constraints.txt \
    --build-constraint /out/provenance/constraints.txt \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --extra-index-url https://flashinfer.ai/whl \
    --requirement /out/provenance/default-requirements.txt \
    --wheel-dir /out/wheels /out/wheels/sglang-*.whl

# Resolve again offline against the resulting wheels; never suppress dependencies.
python3.12 -m pip install --dry-run --ignore-installed --no-index \
    --find-links /out/wheels --constraint /out/provenance/constraints.txt \
    --requirement /out/provenance/default-requirements.txt \
    --report /out/provenance/resolution.json /out/wheels/sglang-*.whl
python3.12 - <<'PY'
import email
import hashlib
import importlib.machinery
import json
import shutil
import zipfile
from pathlib import Path

out = Path("/out")
proof = out / "provenance"
source = json.loads((proof / "source.json").read_text())
wheels = list((out / "wheels").glob("sglang-*.whl"))
assert len(wheels) == 1
wheel = wheels[0]
with zipfile.ZipFile(wheel) as archive:
    metadata = email.message_from_bytes(archive.read(next(
        name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
    )))
    assert metadata["Version"] == source["version"]
    for extension in json.loads((proof / "extensions.json").read_text()):
        stem = extension["module"].replace(".", "/")
        assert any(stem + suffix in archive.namelist()
                   for suffix in importlib.machinery.EXTENSION_SUFFIXES), stem
resolved = json.loads((proof / "resolution.json").read_text())["install"]
versions = {item["metadata"]["name"].lower().replace("_", "-"): item["metadata"]["version"]
            for item in resolved}
for name, version in {
    "torch": "2.14.1+cu130", "flashinfer-python": "0.7.4",
    "flashinfer-cubin": "0.7.4", "sgl-deep-gemm": "0.2.8", "sglang-kernel": "0.4.11",
}.items():
    assert versions.get(name) == version, (name, versions.get(name))
(proof / "runtime-constraints.txt").write_text("".join(
    f"{name}=={version}\n" for name, version in sorted(versions.items())
))
(proof / "wheels.sha256").write_text("".join(
    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n"
    for path in sorted((out / "wheels").glob("*.whl"))
))
(out / "sglang-wheel").mkdir()
shutil.copy2(wheel, out / "sglang-wheel" / wheel.name)
PY
# Test-only additions are solved in build, constrained to the runtime graph.
python3.12 -m pip wheel --wheel-dir /out/test-wheels \
    --constraint /out/provenance/runtime-constraints.txt pytest
rustc -vV > /out/provenance/rustc.txt
cargo --version --verbose > /out/provenance/cargo.txt
nvcc --version > /out/provenance/nvcc.txt
BUILD

FROM base AS runtime
ENV SGLANG_RUST_BUILD_MODE=never \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
COPY --from=build /out/provenance/ /opt/sm120/provenance/
COPY --from=build /out/sglang-wheel/ /opt/sm120/wheels/
COPY source/docker/sm120/entrypoint.sh /usr/local/bin/sm120-entry
RUN --network=none --mount=type=bind,from=build,source=/out/wheels,target=/wheels,ro \
    cd /wheels \
    && sha256sum --check /opt/sm120/provenance/wheels.sha256 \
    && python3.12 -m pip install --no-index --find-links /wheels \
       --constraint /opt/sm120/provenance/runtime-constraints.txt \
       --requirement /opt/sm120/provenance/default-requirements.txt \
       --report /opt/sm120/provenance/runtime-install.json /wheels/sglang-*.whl \
    && python3.12 -m pip check \
    && python3.12 -m pip freeze --all > /opt/sm120/provenance/runtime-freeze.txt \
    && dpkg-query -W > /opt/sm120/provenance/system-packages.txt \
    && bash /usr/local/bin/sm120-entry _verify-image
WORKDIR /work
USER 65532:65532
ENTRYPOINT ["/bin/bash", "/usr/local/bin/sm120-entry", "_inside"]
CMD ["help"]

FROM runtime AS test
USER root
RUN --network=none --mount=type=bind,from=build,source=/out/test-wheels,target=/test-wheels,ro \
    python3.12 -m pip install --no-index --find-links /test-wheels \
       --constraint /opt/sm120/provenance/runtime-constraints.txt \
       --report /opt/sm120/provenance/test-install.json pytest \
    && python3.12 -m pip check \
    && bash /usr/local/bin/sm120-entry _verify-image
USER 65532:65532
