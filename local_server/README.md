# CryoZeta local server

A local-only web interface for running [CryoZeta](https://github.com/kiharalab/CryoZeta)
on your own GPU workstation. Upload a cryo-EM map, paste your sequences, submit
a job, watch it run, download the models.

**Nothing leaves your machine.** No job is ever sent to the Kihara Lab EM server
or to any external inference service. The only optional network access is a CDN
request for the 3D structure viewer, which degrades gracefully when blocked.

This directory is a *wrapper*. It does not modify CryoZeta's modeling code; it
generates CryoZeta's input JSON and invokes the repository's own
`inference_demo.sh` / `large_inference_demo.sh` scripts.

---

## Contents

- [Hardware requirements](#hardware-requirements)
- [Installation](#installation)
- [Starting and stopping the server](#starting-and-stopping-the-server)
- [Preparing inputs](#preparing-inputs)
- [MSA requirements](#msa-requirements)
- [How GPU scheduling works](#how-gpu-scheduling-works)
- [Where jobs and results are stored](#where-jobs-and-results-are-stored)
- [Recovering from interrupted jobs](#recovering-from-interrupted-jobs)
- [Troubleshooting](#troubleshooting)
- [Changing the port](#changing-the-port)
- [Sharing with your lab (Tailscale)](#sharing-with-your-lab-tailscale)
- [Remote access and LAN exposure](#remote-access-and-lan-exposure)
- [Configuration reference](#configuration-reference)
- [Testing](#testing)
- [Licence](#licence)

---

## Hardware requirements

These come from CryoZeta itself, not from this wrapper:

| Requirement | Detail |
|---|---|
| OS | **Linux only.** CryoZeta's Pixi workspace declares `platforms = ["linux-64"]`, with `glibc >= 2.31` and kernel `>= 4.18`. There is no macOS or Windows build. |
| GPU | NVIDIA, CUDA-capable, **32 GB VRAM or more** |
| Driver | Must support CUDA 11.0 or newer (check `nvidia-smi`) |
| Disk | Tens of GB. Weights alone are several GB; each job writes intermediate tensors. |

Capacity, per CryoZeta's own notes: roughly **2,800 residues/nucleotides** in
standard mode. Above that, use large/cycle mode.

Run the preflight to check a machine:

```bash
./local_server/preflight.sh
```

It reports the GPU model, VRAM, compute capability, driver version, the
driver's maximum CUDA version, which Pixi environment will be selected, and
whether the weights, TEASER++ and Pixi environments are installed.

---

## Installation

### 1. Clone this repository

CryoZeta is included as a **git submodule**, so clone recursively. A plain
`git clone` leaves `external/CryoZeta/` empty and nothing will run.

```bash
git clone --recurse-submodules https://github.com/conmmul/cryozeta-local.git
```

If you already cloned without `--recurse-submodules`, fetch it afterwards:

```bash
git submodule update --init --recursive
```

Verify the submodule actually populated — this must print the script, not
"No such file or directory":

```bash
ls cryozeta-local/external/CryoZeta/inference_demo.sh
```

### 2. Check the machine

Before installing anything, confirm the host can run CryoZeta at all:

```bash
cd cryozeta-local/local_server
./preflight.sh
```

This is read-only and changes nothing. It reports the GPU, VRAM, compute
capability, driver version, the maximum CUDA version the driver supports, which
Pixi environment will be selected, and what is still missing — with the exact
command to fix each item. Exit status is 0 when the host is ready.

Expect it to fail on a laptop: CryoZeta is Linux-only and needs a 32 GB NVIDIA
GPU.

### 3. Install CryoZeta itself

The web server does **not** install CryoZeta, and does not modify it. Run the
upstream setup inside the submodule:

```bash
curl -fsSL https://pixi.sh/install.sh | bash
cd cryozeta-local/external/CryoZeta
pixi run setup
```

`pixi run setup` installs dependencies, auto-detects your CUDA version,
downloads the weights and bundled example from Hugging Face, and clones and
builds TEASER++. Budget 15+ minutes and several GB.

> Note: `assets/` is **not** in the git repository. The model weights and the
> bundled example come from Hugging Face via this step, so a fresh clone has no
> weights until you run it.

Confirm CryoZeta works on its own before involving the web server:

```bash
bash external/CryoZeta/inference_demo.sh
```

Then re-run `./preflight.sh` — it should now report **READY**.

### 4. Start the web server

```bash
cd cryozeta-local/local_server
./start_local_server.sh
```

On first run this creates `local_server/.venv`, installs FastAPI/Uvicorn/Jinja2,
creates the data directories, initialises the SQLite database, and starts
Uvicorn. The web dependencies are installed in **their own virtualenv**, never
into CryoZeta's Pixi environment, so upgrading one cannot break the other.

### Finding CryoZeta

The server locates the CryoZeta checkout at runtime, in this order:

1. `CRYOZETA_WEB_REPO`
2. `CRYOZETA_REPO` / `PIXI_PROJECT_ROOT`
3. `../external/CryoZeta` (the submodule in this repository), `../CryoZeta`
4. `./external/CryoZeta`, `./CryoZeta`, `~/CryoZeta`

A directory counts as a CryoZeta checkout only if it contains
`inference_demo.sh`, `large_inference_demo.sh` and a `pyproject.toml` declaring
`name = "cryozeta"`. If yours is elsewhere:

```bash
export CRYOZETA_WEB_REPO=/opt/CryoZeta
```

---

## Starting and stopping the server

```bash
./start_local_server.sh                 # background, http://127.0.0.1:8000
./start_local_server.sh --foreground    # stay attached (Ctrl-C to quit)
./start_local_server.sh --port 8080
./start_local_server.sh --skip-install  # don't touch the virtualenv
./stop_local_server.sh                  # stop the server, leave jobs running
./stop_local_server.sh --kill-jobs      # also terminate running CryoZeta jobs
```

Then open:

> **http://127.0.0.1:8000**

Stopping the server does **not** kill running jobs by default — they are long
and expensive. Those jobs are orphaned, and on the next start they are marked
`interrupted` (see [below](#recovering-from-interrupted-jobs)). Use
`--kill-jobs` if you want them terminated cleanly instead.

### Command line

Every browser action has a CLI equivalent:

```bash
cd local_server
.venv/bin/python -m app.cli preflight          # environment report
.venv/bin/python -m app.cli preflight --json   # machine-readable
.venv/bin/python -m app.cli jobs               # list jobs
.venv/bin/python -m app.cli cancel <job-id>    # cancel a queued job
.venv/bin/python -m app.cli serve              # run in the foreground

# Submit without a browser. --sequence is repeatable:
#   TYPE:SEQUENCE[:COUNT[:MSA_DIR]]   TYPE = protein | dna | rna
.venv/bin/python -m app.cli submit \
    --map /data/emd_44046.map.gz \
    --resolution 2.99 \
    --contour-level 0.3 \
    --sequence "protein:MKTAYIAKQRQ...:2:/data/msa/chainA" \
    --sequence "dna:ACGTACGT:1" \
    --title "my complex" \
    --wait
```

### Running it as a service

`systemd/cryozeta-local-server.service.example` is a **user** service example.
It is not installed or enabled automatically — copy it, edit the paths, and
enable it yourself. Instructions are in the file's header comments.

---

## Preparing inputs

**Map.** `.mrc`, `.map`, `.mrc.gz` or `.map.gz`. The file is checked with
`mrcfile` before the job is queued, so a mislabelled or truncated file is
rejected immediately rather than 20 minutes into a GPU run. Gzipped maps are
passed to CryoZeta compressed (it accepts them); decompression is only used for
validation, with a streaming size cap against decompression bombs.

**Resolution.** In angstroms, 0.5–30.0.

**Contour level.** Must be **non-zero**. CryoZeta thresholds the density grid
with this value, so zero produces a degenerate map. The author-recommended
contour level from EMDB is the right choice.

**Sequences.** One entry per distinct chain, with a copy count. Paste plain
one-letter codes; FASTA headers, whitespace, residue numbering, gap characters
and a trailing `*` are stripped automatically. Alphabets are validated
(protein `ACDEFGHIKLMNPQRSTVWYX`, DNA `ACGTN`, RNA `ACGUN`), and common
DNA/RNA mix-ups are caught explicitly.

Identical sequences of the same type are **merged automatically**: two rows of
the same protein become one entity with the counts summed, sharing one MSA
directory.

**Standard vs large/cycle.** The form recommends large/cycle mode above
**2,800** total residues/nucleotides, the threshold documented in CryoZeta's
README. The recommendation is applied automatically until you change the
selector yourself, and the threshold is configurable
(`CRYOZETA_WEB_LARGE_THRESHOLD`) rather than hard-coded, so it can track
upstream changes.

---

## MSA requirements

**CryoZeta does not bundle MSA generation.** The upstream repository ships no
MSA pipeline, so you must supply precomputed alignments for protein and RNA
chains. Upload a `.zip` per chain, or give an absolute path to a directory on
the server. Nested layouts (`archive/msas/chainA/...`) are handled, and macOS
`__MACOSX` resource forks are ignored.

Required filenames, taken from `src/cryozeta/data/msa_featurizer.py` rather
than from documentation, because the code is stricter than the prose:

| Chain type | Required files |
|---|---|
| **Protein**, complex with **one** distinct protein sequence | `mmseqs_other_hits.a3m` |
| **Protein**, complex with **two or more** distinct protein sequences | `mmseqs_other_hits.a3m` **and** `uniref100_hits.a3m` |
| **RNA** | `rnacentral.a3m` |
| **DNA** | none — DNA needs no MSA |

The pairing rule is subtle and worth stating precisely. CryoZeta computes
`is_homomer_or_monomer = len(set(protein_sequences)) == 1`. **Copy counts are
irrelevant** — eight copies of one sequence is still a homomer and needs no
pairing MSA, while two different sequences with one copy each *does*. When a
pairing MSA is required and missing, CryoZeta fails on a bare `assert` deep in
featurisation. This server checks up front instead and refuses to submit.

Files are also checked for being non-empty and starting with `>`, so an empty
or truncated `.a3m` is caught at submission time.

**Jobs with missing required MSAs are never submitted silently.** The form
lists exactly which files are missing for which chain.

### Generating MSAs

CryoZeta's README recommends:

- **Protein:** [ColabFold search](https://github.com/sokrypton/ColabFold/blob/main/colabfold_search.sh)
- **RNA:** [rMSA](https://github.com/pylelab/rMSA), or the lighter
  [RoseTTAFold2NA helper](https://github.com/uw-ipd/RoseTTAFold2NA/blob/main/input_prep/make_rna_msa.sh)

Rename the outputs to the filenames in the table above.

### Adding an automatic backend later

MSA sourcing sits behind the `MSAProvider` interface in `app/msa.py`:

```python
class MSAProvider(ABC):
    def provide(self, *, sequence, seq_type, needs_pairing, destination) -> Path: ...
```

`UploadedArchiveProvider` and `LocalDirectoryProvider` implement it today.
`RemoteColabFoldProvider` is a deliberate stub: generating alignments through
ColabFold's public endpoint **transmits your sequence to a third party**, which
contradicts this server's local-only guarantee. It must stay an explicit opt-in
(`CRYOZETA_WEB_ALLOW_REMOTE_MSA=1`) and is not implemented. A local
MMseqs2/ColabFold install can be added as another provider without touching the
job pipeline — and without requiring a huge sequence database for the first
working version.

---

## How GPU scheduling works

- The server detects GPUs at startup with `nvidia-smi`. Override with
  `CRYOZETA_WEB_GPUS=0,2`.
- **One worker thread per GPU**, each running at most one job at a time. Two
  jobs can never share a GPU.
- **Different GPUs run jobs concurrently.**
- A job can pin itself to a specific GPU, or leave the selector on "First
  available" and take whichever frees up first.
- Each job's subprocess gets `CUDA_VISIBLE_DEVICES` pinned for the whole
  process group, so nothing the pipeline spawns can wander onto a GPU another
  job owns.
- Jobs are started in their own session (`start_new_session=True`). Cancelling
  signals the entire **process group** with `SIGTERM`, waits a grace period
  (default 20s), then escalates to `SIGKILL`. Signalling only the `bash`
  wrapper would leave the Python process holding the GPU.

The server does not check whether *other* users' processes are already using a
GPU. On a shared machine, restrict it: `CRYOZETA_WEB_GPUS=1`.

---

## Where jobs and results are stored

Default data root `~/cryozeta-web-data`, overridable with
`CRYOZETA_WEB_DATA_ROOT`:

```
~/cryozeta-web-data/
├── cryozeta.sqlite3          job metadata (survives restarts)
├── msa_library/              reserved for shared MSA reuse
├── run/                      server.pid, server.log, upload staging
└── jobs/<uuid>/
    ├── input/                the uploaded map
    ├── msa/<hash>/           one directory per distinct sequence
    ├── spec/input.json       the generated CryoZeta input JSON
    ├── output/               CryoZeta's dump_dir
    │   └── <entry_name>/
    │       ├── CryoZeta-Detection/
    │       ├── CryoZeta/seed_101/predictions/
    │       ├── CryoZeta-Interpolate/
    │       └── CryoZeta-Final/      <-- primary ranked models
    ├── logs/job.log          combined stdout + stderr
    └── meta.json             human-readable submission snapshot
```

Every job directory is a **UUID**. Your job title is never used as a path — it
is sanitised for display, and separately sanitised into a restricted
`entry_name` for the directory CryoZeta creates under `output/`.

**Nothing is ever deleted automatically.** Inputs and results stay until you
remove them yourself.

Large/cycle jobs additionally produce `output/combined.cif`.

---

## Recovering from interrupted jobs

If the server stops while a job is running — crash, reboot, `stop_local_server.sh`
without `--kill-jobs` — the child process is orphaned and its outputs are
partial.

On the next startup those jobs are marked **`interrupted`**, never `completed`.
This is enforced in the state machine: the transition
`interrupted -> completed` does not exist. A job that was cut off can never be
mistaken for a finished one.

To recover, open the job and press **Rerun**. This creates a *new* job that
copies the original inputs, MSAs and generated JSON into a fresh UUID
directory, rewrites the absolute paths inside `input.json` to point at the new
directory, and queues it with `--overwrite` enabled. The original job and its
partial outputs are left untouched.

---

## Troubleshooting

The job page turns CryoZeta's output into a one-line explanation. The
underlying causes and fixes:

### `pixi: command not found` (exit 127)

Pixi is not on the server's `PATH`. It installs to `~/.pixi/bin`, which is
often missing from a service environment.

```bash
export PATH="$HOME/.pixi/bin:$PATH"
```

For systemd, set it in the unit's `Environment=PATH=...` line.

### `environment ... not found` / `--frozen` errors

The demo scripts run `pixi run --no-install --frozen`, which refuses to install
anything implicitly. Install the environment first:

```bash
cd /path/to/CryoZeta
pixi install -e default     # or cu11 / cu13
```

Check which one you need with `./local_server/preflight.sh`.

### `CUDA out of memory`

The complex is too large for the GPU. Options, in order of effort: switch to
large/cycle mode; pick a GPU with more VRAM; reduce the number of chains.
CryoZeta documents ~2,800 residues as the standard-mode ceiling on a 32 GB card.

### `CUDA driver version is insufficient` / `no kernel image is available`

The Pixi environment's CUDA build does not match your driver or GPU
architecture. CryoZeta selects between three environments:

| Compute capability | Driver CUDA | Environment |
|---|---|---|
| >= 10.0 (Blackwell) | >= 13 | `cu13` |
| >= 8.0 (Ampere/Ada/Hopper) | >= 12 | `default` (CUDA 12.8) |
| < 8.0 (Volta/Turing/older) | >= 11 | `cu11` |

Override with `CRYOZETA_WEB_PIXI_ENV=cu11`.

### `No pairing-MSA of ... (please check .../uniref100_hits.a3m)`

Your complex has two or more distinct protein sequences, so every protein chain
needs `uniref100_hits.a3m`. See [MSA requirements](#msa-requirements). The web
form normally catches this before submission — you will only see it if the file
was removed after the job was queued.

### TEASER++ errors (`libteaser.so`, import failures)

```bash
cd /path/to/CryoZeta
pixi run build-teaser
```

It clones and builds TEASER++ under `externals/`. Needs `cmake` and a working
C++ toolchain. If the build fails on a compiler mismatch, check that you are
in the Pixi environment — it pins matching `gcc`/`gxx` versions.

### Compilation errors (`ninja: build stopped`, `nvcc` failures)

CryoZeta JIT-compiles a CUDA layer-norm extension on first use. On a shared or
read-only install, point the cache somewhere writable:

```bash
export CRYOZETA_TORCH_EXTENSIONS_DIR=/scratch/$USER/cryozeta-ext
export TMPDIR=/scratch/$USER/tmp
```

### `No space left on device`

Jobs write large intermediate tensors. Free space, move `CRYOZETA_WEB_DATA_ROOT`
to a bigger volume, and set `TMPDIR` as above — compilation under a full `/tmp`
fails confusingly. The preflight warns below 50 GiB free.

### Checkpoints not found

```bash
cd /path/to/CryoZeta
pixi run download-assets
```

`assets/` is **not** in the git repository; it comes from Hugging Face.

---

## Changing the port

Any of:

```bash
./start_local_server.sh --port 8080
export CRYOZETA_WEB_PORT=8080
```

For a systemd unit, edit both `Environment=CRYOZETA_WEB_PORT=` and the `--port`
argument in `ExecStart=`.

---

## Sharing with your lab (Tailscale)

This is the supported way to let lab members use the server. It publishes over
HTTPS to your private tailnet, without opening a port to the internet and
without a sysadmin, a certificate or a public DNS record.

The app itself **stays bound to 127.0.0.1**. `tailscale serve` terminates TLS
and forwards to it, so the only people who can reach it are members of your
tailnet, whom Tailscale has already authenticated.

### One-time setup

On the GPU workstation:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

In the [Tailscale admin console](https://login.tailscale.com/admin/dns), enable
**MagicDNS** and **HTTPS Certificates** (DNS → HTTPS Certificates). `tailscale
serve` cannot issue an HTTPS URL without this.

Then invite lab members to the tailnet. Each installs Tailscale and signs in;
no further per-person setup is needed.

### Starting it

```bash
./start_local_server.sh --tailscale
```

The script verifies Tailscale is installed and connected, starts the server on
loopback, publishes it, and prints the URL, which looks like:

> **https://gpu-workstation.tailnet-name.ts.net**

Anyone on the tailnet can open that from a laptop, at home or on campus.

To stop publishing (the server keeps running locally):

```bash
tailscale serve --https=443 off
```

`./stop_local_server.sh` withdraws it automatically.

### What lab members see

Jobs become **attributed**. The submitter's Tailscale login is recorded and
shown in the header, in a "Submitted by" column on the Jobs page, and on each
job's detail page. Existing databases are upgraded in place and old jobs simply
show a blank submitter.

### Understand the trust model before you rely on it

| Property | Behaviour |
|---|---|
| Who can reach it | Anyone on your tailnet, and nobody else |
| Who can submit jobs | Anyone on your tailnet |
| Who can see results | **Everyone** on your tailnet, including other people's jobs and uploaded maps |
| Who can cancel jobs | **Anyone** on your tailnet, including other people's jobs |
| Identity | Recorded for attribution; it is **not** an access control |

There are still no per-user permissions. Identity answers "who ran this?", not
"who is allowed to?". **The tailnet boundary is the security control** — treat
tailnet membership as equivalent to a shell account on the workstation, because
in practice submitting a job runs code on it.

If you need per-user isolation or quotas, that is a larger change than this
wrapper makes; say so and it can be designed.

> ### ⚠️ Do not use Tailscale Funnel
>
> `tailscale funnel` publishes to the **public internet**. Because this server
> has no authentication, that would let anyone who finds the hostname upload
> files, read every result and consume your GPUs. Use `tailscale serve`, which
> is tailnet-only, and which is what `--tailscale` configures.

### How identity is established, and why it cannot be spoofed

`tailscale serve` injects a `Tailscale-User-Login` header. That header is only
believed when **both** hold:

1. the operator explicitly enabled Tailscale mode
   (`CRYOZETA_WEB_TRUST_TAILSCALE_HEADERS=1`, set by `--tailscale`), and
2. the request arrived from loopback, i.e. from the local proxy.

Without both, the header is ignored — otherwise anyone who could reach the port
could impersonate a colleague by setting a header themselves. When the app is
instead bound directly to the tailnet address, identity comes from `tailscale
whois` on the real peer IP rather than from any header.


## Remote access and LAN exposure

The server binds **127.0.0.1** by default and **refuses to start on any other
address** unless you explicitly opt in. Both `start_local_server.sh` and the
application itself enforce this.

### For a single user: SSH tunnel

To share with a lab, prefer
[Tailscale](#sharing-with-your-lab-tailscale). For just yourself:

The GPU workstation is usually not the machine you browse from. Do not expose
the server — forward the port instead:

```bash
ssh -N -L 8000:127.0.0.1:8000 you@gpu-workstation
```

Then open **http://127.0.0.1:8000** on your laptop. The server stays bound to
loopback on the workstation, and the traffic is encrypted and authenticated by
SSH.

### If you must bind a network interface

> ### ⚠️ Security warning
>
> **This application has no authentication of any kind.** There are no
> accounts, no passwords, no authorisation checks, and no audit log. Anyone who
> can reach the port can:
>
> - read every job, every uploaded map and every result on the server
> - upload arbitrary files and consume all your GPU capacity
> - read any file inside a job directory
>
> The file-download endpoint is confined to job directories and rejects path
> traversal, absolute paths and symlinks — but that is containment, not access
> control. There is no protection against a person who can simply open the URL.
>
> Only ever do this on a trusted, firewalled network, and prefer the SSH tunnel.

```bash
export CRYOZETA_WEB_ALLOW_LAN=1
./start_local_server.sh --host 0.0.0.0
```

Without `CRYOZETA_WEB_ALLOW_LAN=1`, a non-loopback `--host` is refused with an
explanation.

---

## Configuration reference

All settings are environment variables prefixed `CRYOZETA_WEB_`.

| Variable | Default | Meaning |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind address |
| `PORT` | `8000` | Bind port |
| `ALLOW_LAN` | `0` | Required to bind a non-loopback address |
| `DATA_ROOT` | `~/cryozeta-web-data` | Where jobs and the database live |
| `REPO` | auto-detected | Path to the CryoZeta checkout |
| `PIXI_ENV` | auto-detected | Force `default`, `cu11` or `cu13` |
| `GPUS` | auto-detected | e.g. `0,2` — restrict schedulable GPUs |
| `MAX_UPLOAD_MB` | `4096` | Map upload limit |
| `MAX_DECOMPRESSED_MB` | `16384` | Decompressed size cap |
| `MAX_MSA_ARCHIVE_MB` | `2048` | MSA ZIP upload limit |
| `MAX_ARCHIVE_MEMBERS` | `10000` | Max files in an MSA archive |
| `MAX_COMPRESSION_RATIO` | `200` | Decompression-bomb guard |
| `MAX_TOTAL_SEQ_LEN` | `20000` | Rejects absurd submissions |
| `LARGE_THRESHOLD` | `2800` | Large/cycle recommendation threshold |
| `CANCEL_GRACE_SECONDS` | `20` | SIGTERM → SIGKILL grace period |
| `TRUST_TAILSCALE_HEADERS` | `0` | Trust `tailscale serve` identity headers from loopback |
| `ALLOW_REMOTE_MSA` | `0` | Reserved for a future external MSA backend |

---

## Testing

```bash
./run_tests.sh              # unit + integration, no GPU needed
./run_tests.sh --smoke      # additionally run the real bundled example
./run_tests.sh -k msa       # forward arguments to pytest
```

The default run needs **no GPU, no weights and no pixi**. The inference binary
is replaced by `tests/fake/fake_inference_demo.sh`, which accepts the same
flags as `inference_demo.sh`, asserts the shape of the generated JSON (exiting
non-zero if it is wrong), and writes an output tree with the same layout. A
passing integration test is therefore a real check that our JSON matches
CryoZeta's contract.

Coverage: sequence normalisation and de-duplication, JSON generation, MSA
requirement rules and validation, archive extraction (traversal, symlinks,
bombs, member limits), path security, job-state transitions and crash
recovery, command construction for both pipelines, failure-message
classification, and the full submit → schedule → run → results → download
workflow including cancellation and rerun.

`tests/smoke/` holds the real-GPU tests. They skip themselves automatically
unless the host is genuinely ready, and they verify both that our generated
JSON structurally matches the bundled `assets/examples/example.json` and that
the bundled example runs end to end through this server.

---

## Licence

CryoZeta itself is dual-licensed, and the two halves are **not** under the same
terms:

- **Source code** — [GNU General Public License v3.0](https://github.com/kiharalab/CryoZeta/blob/main/LICENSE).
- **Trained model weights** — a **separate** licence, free for **academic and
  non-commercial research use only**. Commercial use is **not permitted**
  without permission from the authors. See
  [`WEIGHT_LICENSE.md`](https://github.com/kiharalab/CryoZeta/blob/main/WEIGHT_LICENSE.md).

Running this server locally does not change either licence. If you are at a
commercial organisation, the weights restriction applies to you regardless of
where inference happens.

This wrapper is a derivative work of a GPL-3.0 project and is distributed under
the same terms.

If you use CryoZeta, cite the paper, and also cite
[Protenix](https://github.com/bytedance/Protenix) and
[OpenFold](https://github.com/aqlaboratory/openfold), on which it is built.
