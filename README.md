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

That prints the GPU, VRAM, compute capability, driver, the CUDA version your
driver supports, which Pixi environment will be used, and whether the weights
and TEASER++ are installed — and tells you how to fix anything missing.

## Set up

Install CryoZeta itself (downloads several GB of model weights and builds
TEASER++, ~15 minutes):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
cd cryozeta-local/external/CryoZeta && pixi run setup
```

Then start the web server:

```bash
cd cryozeta-local/local_server && ./start_local_server.sh
```

Open **http://127.0.0.1:8000**

## Sharing it with a lab

```bash
./start_local_server.sh --tailscale
```

Publishes over HTTPS to your private Tailscale network, so lab members can use
it from their own laptops. It is **not** exposed to the public internet, and
the app itself stays bound to loopback.

Read the security model first: anyone on the tailnet can see every result and
cancel any job. See
[local_server/README.md](local_server/README.md#sharing-with-your-lab-tailscale).

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
