#!/usr/bin/env bash
set -euo pipefail
export GIT_OPTIONAL_LOCKS=0
export GIT_NO_LAZY_FETCH=1

die() { printf 'sm120: %s\n' "$*" >&2; exit 1; }

usage() {
    cat <<'HELP'
Usage (Linux host, non-root user; clean, reviewed sm120 commit required):
  bash docker/sm120/entrypoint.sh build <full-commit> [build-id]
  bash docker/sm120/entrypoint.sh test  <full-commit> <build-id> [pytest nodeids...]
  bash docker/sm120/entrypoint.sh serve <full-commit> <build-id> [sglang arguments...]
  bash docker/sm120/entrypoint.sh logs  <full-commit> <build-id>
  bash docker/sm120/entrypoint.sh stop  <full-commit> <build-id>

Paths: SM120_SOURCE=/data/yyang/sglang
       SM120_WORK=/data/yyang/sglang-work
       SM120_MODEL=/data/yyang/models/MoonMath-ai_DeepSeek-V4.1-Flash-1017-uncensored
Build never installs host dependencies. It requires bash, git, python3.12,
timeout, and an already usable Docker engine/NVIDIA container runtime.
Docker layers and BuildKit cache remain in the existing daemon storage; only
source copies, exported evidence, temporary files and logs use SM120_WORK.

Base/frontend pulls: daemon networking, not client HTTP_PROXY or SM120_BUILD_PROXY.
Optional SM120_CUDA_BASE=nvidia/cuda@sha256:<verified-digest> uses an already
local official base instead of pulling its tag. CUDA13 devel/Ubuntu24.04 is
checked in the build. No digest is supplied or claimed verified here.

Optional build-only proxy (TLS verification remains enabled):
  SM120_BUILD_PROXY=http://<container-reachable-proxy>:7890 ... build ...
For an EXISTING Linux daemon-host listener at 127.0.0.1:7890 only:
  SM120_BUILD_NETWORK=host SM120_BUILD_PROXY=http://127.0.0.1:7890 ... build ...
This opts build RUN instructions into host networking, NOT test/serve or pulls.
If daemon pulls or host-network entitlement fail, stop: no daemon/global proxy
changes, tunnel creation, listener replacement, or automatic retries here.
Do not put proxy credentials in the URL. No proxy enters test/serve.

Serve is foreground, TP4, 8192-token smoke only. DSpark and long-context
acceptance are separate; no permanent speculative/attention disabling is set.
After git pull, build a new wheel/images for the new full commit before use.
HELP
}

verify_image() {
    python3.12 - <<'PY'
import hashlib
import importlib
import importlib.metadata as metadata
import json
import sysconfig
import zipfile
from pathlib import Path

import torch

root = Path("/opt/sm120")
proof = root / "provenance"
source = json.loads((proof / "source.json").read_text())
assert metadata.version("sglang") == source["version"]
assert str(torch.__version__) == "2.14.1+cu130"
assert torch.version.cuda == "13.0"
for line in (proof / "runtime-constraints.txt").read_text().splitlines():
    name, version = line.split("==", 1)
    assert metadata.version(name) == version, name
wheel, = (root / "wheels").glob("sglang-*.whl")
expected = dict(line.split("  ", 1)[::-1]
                for line in (proof / "wheels.sha256").read_text().splitlines())
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == expected[wheel.name]
site = Path(sysconfig.get_path("platlib")).resolve()
distribution = metadata.distribution("sglang")
with zipfile.ZipFile(wheel) as archive:
    for name in archive.namelist():
        # Compare actual installed package bytes, not only an image label/version.
        if name.startswith("sglang/") and not name.endswith("/"):
            installed = Path(distribution.locate_file(name)).resolve()
            assert installed.is_relative_to(site), installed
            assert hashlib.sha256(installed.read_bytes()).digest() == hashlib.sha256(
                archive.read(name)
            ).digest(), name
for extension in json.loads((proof / "extensions.json").read_text()):
    if fingerprint := extension.get("torch_build"):
        assert fingerprint["torch_version"] == str(torch.__version__)
        assert fingerprint["torch_cuda"] == torch.version.cuda
        assert fingerprint["torch_cxx11_abi"] == bool(torch.compiled_with_cxx11_abi())
    module = importlib.import_module(extension["module"])
    assert Path(module.__file__).resolve().is_relative_to(site), module.__file__
print(json.dumps({"verified_commit": source["commit"], "version": source["version"],
                  "sglang_wheel_sha256": expected[wheel.name]}, sort_keys=True))
PY
}

check_source() {
    local directory=$1 expected=$2 actual branch
    [[ -d "$directory/.git" ]] || die "a standalone Git checkout is required: $directory"
    actual=$(git -c safe.directory="$directory" -C "$directory" rev-parse HEAD)
    branch=$(git -c safe.directory="$directory" -C "$directory" symbolic-ref --short HEAD)
    [[ "$actual" == "$expected" && "$branch" == sm120 ]] || die "source commit/branch mismatch"
    [[ -z "$(git -c safe.directory="$directory" -C "$directory" status --porcelain=v1 --untracked-files=all)" ]] \
        || die "source is not clean; review/commit changes first"
    # Reject tracked paths inside private directories/files and submodules by
    # path/mode only, without opening their contents.
    git -c safe.directory="$directory" -C "$directory" ls-files --stage -z | python3.12 -c '
import sys
for entry in sys.stdin.buffer.read().split(b"\0"):
    if not entry:
        continue
    header, path = entry.split(b"\t", 1)
    if b"NOTE.md" in path.split(b"/") or header.startswith(b"160000 "):
        raise SystemExit("private tracked file/submodule: cannot make a complete offline context")
'
}

inside() {
    local mode=${1:-help} expected=${2:-}
    [[ "$mode" == test || "$mode" == serve ]] || { usage; return; }
    [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || die "full expected commit required"
    shift 2
    [[ "$(id -u)" != 0 && "$(id -g)" != 0 ]] || die "test/serve require a non-root UID/GID"
    check_source /src "$expected"
    python3.12 - "$expected" <<'PY'
import json
import sys
from pathlib import Path

source = json.loads(Path("/opt/sm120/provenance/source.json").read_text())
assert source["commit"] == sys.argv[1], "installed wheel is stale; rebuild the images"
PY
    verify_image
    python3.12 - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 4, "exactly GPUs 0-3 must be exposed"
assert all(torch.cuda.get_device_capability(i) == (12, 0) for i in range(4)), "SM120 required"
PY
    cd /work
    if [[ "$mode" == test ]]; then
        if [[ $# == 0 ]]; then
            set -- \
                /src/test/registered/rust/test_rust_extension.py::TestRustExtension::test_bundled_wheel_extension_never_touches_source_or_cargo \
                /src/test/registered/rust/test_rust_extension.py::TestRustExtension::test_bundled_named_variant_never_touches_source_or_cargo \
                /src/test/registered/rust/test_rust_extension.py::TestRustExtension::test_torch_build_configuration_is_versioned_and_relocatable \
                /src/test/registered/unit/mem_cache/test_tree_core_registry.py \
                /src/test/registered/unit/mem_cache/test_rust_tree_core_integration.py::test_default_backend_constructs_real_rust_cpu_cache \
                /src/test/registered/unit/mem_cache/test_rust_tree_core_integration.py::test_insert_then_match_back_returns_the_exact_indices \
                /src/test/registered/unit/mem_cache/test_rust_tree_core_integration.py::test_cuda_core_resolves_the_current_device \
                /src/test/registered/unit/layers/test_dsv41_paged_decode.py \
                /src/test/registered/kernels/ops/attention/test_flash_mla_backends.py \
                /src/test/registered/kernels/ops/attention/test_fp4_indexer.py::test_sm120_full_topk_native_planner_replay \
                /src/test/registered/kernels/ops/attention/test_fp4_indexer.py::test_sm120_early_verify_chunked_variants_replay
        fi
        exec python3.12 -m pytest --import-mode=importlib \
            -o cache_dir=/work/cache/pytest --junitxml=/work/log/pytest.xml -ra "$@"
    fi
    [[ -f /model/config.json ]] || die "local model config missing"
    exec /opt/venv/bin/sglang serve --model-path /model --trust-remote-code \
        --host 0.0.0.0 --port 18000 --tp-size 4 --context-length 8192 \
        --ep-size 2 --engram-config '{"placement":"cpu"}' --kv-cache-dtype fp8_e4m3 \
        --download-dir /work/cache/huggingface "$@"
}

case "${1:-help}" in
    _verify-image) verify_image; exit ;;
    _inside) shift; inside "$@"; exit ;;
    help|-h|--help) usage; exit ;;
    build|test|serve|logs|stop) mode=$1; shift ;;
    *) usage; exit 2 ;;
esac

[[ $# -ge 1 && "$1" =~ ^[0-9a-f]{40}$ ]] || die "supply the reviewed full commit"
commit=$1
shift
build_id=${1:-}
if [[ -z "$build_id" && "$mode" == build ]]; then
    build_id=$(date -u +%Y%m%dt%H%M%sz)
fi
[[ "$build_id" =~ ^[a-z0-9][a-z0-9_.-]{0,47}$ ]] || die "supply a safe build-id"
if [[ $# -gt 0 ]]; then shift; fi
[[ "$mode" != build || $# == 0 ]] || die "build takes only commit and optional build-id"
[[ "$mode" != logs && "$mode" != stop ]] || [[ $# == 0 ]] || die "$mode takes only commit and build-id"
[[ "$(uname -s)" == Linux && "$(id -u)" != 0 && "$(id -g)" != 0 ]] \
    || die "use a non-root Linux host UID/GID"
for tool in git python3.12 docker timeout; do
    command -v "$tool" >/dev/null || die "missing host tool: $tool (install it yourself)"
done

source_dir=${SM120_SOURCE:-/data/yyang/sglang}
work_dir=${SM120_WORK:-/data/yyang/sglang-work}
model_dir=${SM120_MODEL:-/data/yyang/models/MoonMath-ai_DeepSeek-V4.1-Flash-1017-uncensored}
for path in "$source_dir" "$work_dir" "$model_dir"; do
    [[ "$path" == /* && "$path" != *','* && "$path" != *$'\n'* ]] || die "absolute, comma-free paths required"
done
source_dir=$(realpath "$source_dir")
work_dir=$(realpath -m "$work_dir")
model_dir=$(realpath -m "$model_dir")
[[ "$work_dir/" != "$source_dir/"* && "$source_dir/" != "$work_dir/"* ]] \
    || die "source and work must not contain one another"
[[ "$work_dir/" != "$model_dir/"* && "$model_dir/" != "$work_dir/"* ]] \
    || die "model and work must not contain one another"
check_source "$source_dir" "$commit"
mkdir -p "$work_dir"/{build,cache,tmp,log,home}
job="$work_dir/build/$commit/$build_id"
image="sglang-sm120-yyang:$commit-$build_id"
test_image="sglang-sm120-yyang:test-$commit-$build_id"
name="sglang-sm120-yyang-$mode-${commit:0:12}-$build_id"

if [[ "$mode" == build ]]; then
    [[ "$(realpath "${BASH_SOURCE[0]}")" == "$source_dir/docker/sm120/entrypoint.sh" ]] \
        || die "invoke the committed entrypoint from SM120_SOURCE"
    mkdir -p "$work_dir/build/$commit"
    mkdir "$job" || die "build-id already exists; choose a new one (nothing removed)"
    timeout 30 docker info --format '{{.DockerRootDir}}' > "$job/daemon-storage.txt"
    printf 'Docker layers/BuildKit cache stay in daemon storage: %s\n' "$(cat "$job/daemon-storage.txt")"
    base=${SM120_CUDA_BASE:-}
    if [[ -z "$base" ]]; then
        tag=nvidia/cuda:13.0.3-cudnn-devel-ubuntu24.04
        if ! timeout "${SM120_PULL_TIMEOUT:-900}" docker pull --platform linux/amd64 "$tag" \
            > "$job/base-pull.log" 2>&1; then
            die "base pull failed; inspect $job/base-pull.log. Build proxy cannot fix daemon networking. Supply an already-local verified SM120_CUDA_BASE or ask the daemon administrator; no config changed."
        fi
        base=$(timeout 30 docker image inspect "$tag" --format '{{json .RepoDigests}}' | python3.12 -c '
import json, re, sys
matches = [value for value in json.load(sys.stdin)
           if re.fullmatch(r"nvidia/cuda@sha256:[0-9a-f]{64}", value)]
assert len(matches) == 1, "unambiguous official base digest required"
print(matches[0])')
    fi
    [[ "$base" =~ ^nvidia/cuda@sha256:[0-9a-f]{64}$ ]] || die "official nvidia/cuda digest required"
    [[ "$(timeout 30 docker image inspect "$base" --format '{{.Os}}/{{.Architecture}}')" == linux/amd64 ]] \
        || die "the verified base must already be local and linux/amd64"
    printf '%s\n' "$base" > "$job/cuda-base.txt"

    network=${SM120_BUILD_NETWORK:-default}
    [[ "$network" == default || "$network" == host ]] || die "build network must be default or host"
    proxy=${SM120_BUILD_PROXY:-}
    python3.12 - "$proxy" "$network" <<'PY'
import sys
from urllib.parse import urlsplit

if sys.argv[1]:
    proxy = urlsplit(sys.argv[1])
    assert proxy.scheme in ("http", "https") and proxy.hostname and proxy.port
    assert not proxy.username and not proxy.password, "proxy credentials not accepted"
    if proxy.hostname in ("127.0.0.1", "localhost", "::1"):
        assert sys.argv[2] == "host", "container loopback is not host loopback; opt into build host networking"
PY
    proxy_args=()
    for key in HTTP_PROXY HTTPS_PROXY http_proxy https_proxy; do
        proxy_args+=(--build-arg "$key=$proxy")
    done
    for key in ALL_PROXY all_proxy FTP_PROXY ftp_proxy NO_PROXY no_proxy; do
        proxy_args+=(--build-arg "$key=")
    done

    # Local clone only: no fetch, no tag/history/BASE ancestry requirement.
    mkdir "$job/context"
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null GIT_LFS_SKIP_SMUDGE=1 \
        timeout 300 git clone --local --no-hardlinks --single-branch --no-tags \
        --branch sm120 -- "$source_dir" "$job/context/source"
    check_source "$job/context/source" "$commit"
    cp "$job/context/source/docker/sm120/Dockerfile" "$job/context/Dockerfile"
    timestamp=$(git -C "$job/context/source" show -s --format=%ct HEAD)
    version="0.dev$timestamp+sm120.g$commit"
    args=(--platform linux/amd64 --progress plain --network "$network"
        --file "$job/context/Dockerfile" --build-arg "CUDA_BASE=$base"
        --build-arg "SOURCE_COMMIT=$commit" --build-arg "SOURCE_VERSION=$version"
        --build-arg UPSTREAM_BASE=39af8b1211791c9e6936afad499b01f36ecc4b9c
        --label io.sglang.sm120.owner=yyang --label "org.opencontainers.image.revision=$commit")
    for ref in "$image" "$test_image"; do
        if timeout 30 docker image inspect "$ref" >/dev/null 2>&1; then die "image already exists: $ref"; fi
    done
    for target in runtime test; do
        ref=$image
        if [[ "$target" == test ]]; then ref=$test_image; fi
        if ! timeout --foreground "${SM120_BUILD_TIMEOUT:-14400}" docker build \
            "${args[@]}" "${proxy_args[@]}" --target "$target" --tag "$ref" "$job/context" \
            2>&1 | tee "$work_dir/log/build-$commit-$build_id-$target.log"; then
            die "build failed; see the log above. Dependency conflicts are fatal. RUN proxy does not fix daemon/frontend pulls; host-network entitlement/config changes require the administrator."
        fi
    done
    # Export actual wheel and resolver/build evidence without starting a container.
    evidence_name="sglang-sm120-yyang-evidence-${commit:0:12}-$build_id"
    cid=$(timeout 30 docker create --name "$evidence_name" --network none --read-only \
        --user "$(id -u):$(id -g)" --label io.sglang.sm120.owner=yyang \
        --entrypoint /bin/true "$image")
    timeout 120 docker cp "$cid:/opt/sm120" "$job/artifacts"
    timeout 30 docker rm "$cid" >/dev/null
    timeout 30 docker image inspect "$image" "$test_image" > "$job/images.json"
    python3.12 - "$job" "$commit" "$image" "$test_image" <<'PY'
import json
import sys
from pathlib import Path

job = Path(sys.argv[1])
source = json.loads((job / "artifacts/provenance/source.json").read_text())
assert source["commit"] == sys.argv[2]
images = json.loads((job / "images.json").read_text())
(job / "receipt.json").write_text(json.dumps({
    "commit": sys.argv[2], "runtime": sys.argv[3], "test": sys.argv[4],
    "runtime_id": images[0]["Id"], "test_id": images[1]["Id"],
    "version": source["version"], "cuda_base": source["cuda_base"],
}, indent=2) + "\n")
PY
    printf 'Built %s\nEvidence: %s\nNext: bash %q test %s %s\n' \
        "$build_id" "$job" "$source_dir/docker/sm120/entrypoint.sh" "$commit" "$build_id"
    exit
fi

[[ -f "$job/receipt.json" ]] || die "no successful build receipt; rebuild this exact commit first"
read -r runtime_id test_id < <(python3.12 - "$job/receipt.json" "$commit" "$image" "$test_image" <<'PY'
import json
import re
import sys

receipt = json.load(open(sys.argv[1]))
assert (receipt["commit"], receipt["runtime"], receipt["test"]) == tuple(sys.argv[2:])
assert all(re.fullmatch(r"sha256:[0-9a-f]{64}", receipt[key]) for key in ("runtime_id", "test_id"))
print(receipt["runtime_id"], receipt["test_id"])
PY
)
selected=$runtime_id
if [[ "$mode" == test ]]; then selected=$test_id; fi
[[ "$(timeout 30 docker image inspect "$selected" --format '{{index .Config.Labels "io.sglang.sm120.owner"}}')" == yyang ]] \
    || die "not an image owned by this entrypoint"
record_prefix="$job/serve-$build_id-"
if [[ "$mode" == logs || "$mode" == stop ]]; then
    shopt -s nullglob
    records=("$job"/serve-"$build_id"-*.cid)
    shopt -u nullglob
    [[ ${#records[@]} == 1 ]] || die "expected exactly one recorded serve container for this commit/build-id"
    serve_record=${records[0]}
    cid=$(<"$serve_record")
    [[ "$cid" =~ ^[0-9a-f]{64}$ ]] || die "invalid recorded container ID"
    inspected=$(timeout 30 docker container inspect "$cid" --format '{{.Id}} {{.Image}} {{index .Config.Labels "io.sglang.sm120.owner"}} {{index .Config.Labels "io.sglang.sm120.commit"}}' 2>/dev/null) \
        || die "recorded container no longer exists"
    [[ "$inspected" == "$cid $runtime_id yyang $commit" ]] || die "recorded container identity mismatch"
    if [[ "$mode" == logs ]]; then exec docker logs --follow "$cid"; fi
    timeout 30 docker stop --time 20 "$cid"
    rm -f "$serve_record"
    exit
fi
[[ -d "$model_dir" ]] || die "local model directory missing"
model_dir=$(realpath "$model_dir")
run=(--init --label io.sglang.sm120.owner=yyang --label "io.sglang.sm120.commit=$commit"
    --user "$(id -u):$(id -g)" --cap-drop ALL --security-opt no-new-privileges:true
    --read-only --network bridge --gpus '"device=0,1,2,3"'
    --cpuset-cpus 0-31,64-95 --cpuset-mems 0 --memory 400g --memory-swap 400g
    --ulimit memlock=-1:-1 --ipc private --shm-size 32g --workdir /work
    --mount "type=bind,src=$source_dir,dst=/src,readonly"
    --mount "type=bind,src=$model_dir,dst=/model,readonly"
    --mount "type=bind,src=$work_dir,dst=/work"
    --env NVIDIA_VISIBLE_DEVICES=0,1,2,3 --env CUDA_VISIBLE_DEVICES=0,1,2,3
    --env PYTHONPATH= --env PYTHONNOUSERSITE=1 --env PYTHONSAFEPATH=1)
# Override Docker-client proxy defaults too; never carry build proxies to service.
for key in HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY NO_PROXY http_proxy https_proxy all_proxy ftp_proxy no_proxy; do
    run+=(--env "$key=")
done
if [[ "$mode" == serve ]]; then
    run+=(--publish 127.0.0.1:18000:18000)
    serve_record=$(mktemp "$job/serve-$build_id-XXXXXX.cid")
fi
cidfile=$(mktemp "$work_dir/tmp/cid-$mode-$build_id-XXXXXX")
created_cid=
cleanup_done=0
cleanup_run() {
    local status=$?
    (( cleanup_done == 0 )) || return 0
    cleanup_done=1
    if [[ -n "$created_cid" ]]; then
        timeout 30 docker stop --time 20 "$created_cid" >/dev/null 2>&1 || true
        timeout 30 docker rm -f "$created_cid" >/dev/null 2>&1 || true
    fi
    rm -f "$cidfile"
    [[ -z "${serve_record:-}" ]] || rm -f "$serve_record"
    return "$status"
}
trap cleanup_run EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "$mode" == test ]]; then
    if timeout --foreground "${SM120_TEST_TIMEOUT:-3600}" docker run --cidfile "$cidfile" "${run[@]}" \
        "$selected" test "$commit" "$@" 2>&1 | tee "$work_dir/log/test-$commit-$build_id.log"; then
        created_cid=$(<"$cidfile") || die "docker run succeeded but left no readable cidfile"
        [[ "$created_cid" =~ ^[0-9a-f]{64}$ ]] || die "docker returned an invalid container ID"
        exit 0
    else
        exit $?
    fi
fi
if ! docker run --cidfile "$cidfile" -d "${run[@]}" "$selected" serve "$commit" "$@" \
    > "$work_dir/log/serve-$commit-$build_id.log" 2>&1; then
    die "container creation failed; no container cleanup attempted; see $work_dir/log/serve-$commit-$build_id.log"
fi
created_cid=$(<"$cidfile") || die "docker run -d succeeded but left no readable cidfile; a container may still be running"
[[ "$created_cid" =~ ^[0-9a-f]{64}$ ]] || die "docker returned an invalid container ID"
inspected=$(timeout 30 docker container inspect "$created_cid" --format '{{.Id}} {{.Image}} {{index .Config.Labels "io.sglang.sm120.owner"}} {{index .Config.Labels "io.sglang.sm120.commit"}}')
[[ "$inspected" == "$created_cid $runtime_id yyang $commit" ]] || die "created container identity mismatch"
printf '%s\n' "$created_cid" > "$serve_record"
for attempt in $(seq 1 "${SM120_READY_ATTEMPTS:-120}"); do
    if python3.12 - "$attempt" <<'PY'
import sys
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen("http://127.0.0.1:18000/ready", timeout=3) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except (urllib.error.URLError, TimeoutError):
    raise SystemExit(1)
PY
    then
        printf 'Container: %s\nEndpoint: http://127.0.0.1:18000\nLogs: %q logs %s %s\nStop: %q stop %s %s\n' \
            "$created_cid" "$source_dir/docker/sm120/entrypoint.sh" "$commit" "$build_id" \
            "$source_dir/docker/sm120/entrypoint.sh" "$commit" "$build_id"
        cleanup_done=1
        trap - EXIT INT TERM
        exit 0
    fi
    running=$(timeout 15 docker container inspect "$created_cid" --format '{{.State.Running}}' 2>/dev/null || true)
    [[ "$running" == true ]] || die "serve container exited before readiness; inspect $work_dir/log/serve-$commit-$build_id.log"
done
die "serve readiness timed out after ${SM120_READY_ATTEMPTS:-120} probes; inspect $work_dir/log/serve-$commit-$build_id.log"
