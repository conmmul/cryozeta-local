# cryozeta-local

A local web interface for running [CryoZeta](https://github.com/kiharalab/CryoZeta)
on your own GPU workstation. Upload a cryo-EM map, enter your sequences, submit
a job, watch it run, download the models.

Jobs execute on **your** hardware. Nothing is sent to the Kihara Lab EM server
or to any external inference service.

```
cryozeta-local/
├── external/CryoZeta/   upstream CryoZeta (git submodule, unmodified)
└── local_server/        the web application
```

## Clone

This repository contains CryoZeta as a **git submodule**, so it must be cloned
recursively. A plain `git clone` leaves `external/CryoZeta/` empty and nothing
will run.

```bash
git clone --recurse-submodules https://github.com/conmmul/cryozeta-local.git
```

Already cloned without it? Fetch the submodule after the fact:

```bash
git submodule update --init --recursive
```

## Requirements

CryoZeta is **Linux-only** and needs an NVIDIA GPU with **32 GB VRAM or more**.
Its Pixi workspace declares `platforms = ["linux-64"]`, so it cannot be
installed on macOS or Windows. Check any machine with:

```bash
cd cryozeta-local/local_server && ./preflight.sh
```

A 24 GB card (RTX 4090) works, but caps complex size below upstream's
~2,800-residue figure; see the notes in
[local_server/README.md](local_server/README.md#hardware-requirements).

That prints the GPU, VRAM, compute capability, driver, the CUDA version your
driver supports, which Pixi environment will be used, and whether the weights
and TEASER++ are installed — and tells you how to fix anything missing.

## Set up

One command does everything — installs pixi, installs the CUDA environment for
your GPU, downloads the model weights, builds TEASER++, and sets up the web
server:

```bash
cd cryozeta-local/local_server && ./setup.sh
```

Expect 15–40 minutes and several GB on a first run. It is safe to re-run: every
step skips work that is already done, so an interrupted install resumes.

When it prints `Result: READY`, start the server:

```bash
./start_local_server.sh
```

Open **http://127.0.0.1:8000**

## Sharing it with a lab

Three options, documented in
[local_server/README.md](local_server/README.md#sharing-with-your-lab):

- **SSH tunnel** — nothing to configure on the server
- **VPN interface + passphrase** — `app.cli set-password`, then bind to your VPN address
- **Tailscale userspace mode** — `./publish_tailscale.sh start`, an HTTPS URL for the tailnet

If you reach the server through a VPN, **do not run plain `tailscale up`**: it
edits the routing table and `/etc/resolv.conf` and can cut your own SSH access.
`publish_tailscale.sh` uses userspace mode, which cannot touch either.

Binding anywhere other than loopback requires a passphrase. Note that none of
these provide per-user permissions: anyone who gets in can see every result and
cancel any job.

## Documentation

Everything else — MSA requirements, GPU scheduling, where results are stored,
recovering interrupted jobs, CUDA/Pixi/TEASER++ troubleshooting, changing the
port — is in **[local_server/README.md](local_server/README.md)**.

## Licence

CryoZeta's **source code** is GPL-3.0. Its **model weights** are under a
separate licence and are free for **academic and non-commercial research use
only** — commercial use requires permission from the authors. See
[WEIGHT_LICENSE.md](https://github.com/kiharalab/CryoZeta/blob/main/WEIGHT_LICENSE.md).

Running this locally does not change either licence.
