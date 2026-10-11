# TrellisMCP

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Linux or Windows with WSL 2](https://img.shields.io/badge/platform-Linux%20%7C%20WSL%202-blue.svg)
![NVIDIA GPU required](https://img.shields.io/badge/GPU-NVIDIA-76B900.svg)

Give an AI agent a picture and get back a textured 3D model. This is a small
[Model Context Protocol](https://modelcontextprotocol.io) server that runs Microsoft's
[TRELLIS](https://github.com/microsoft/TRELLIS) on your own NVIDIA card, so an agent such as
Claude Code or Codex can make a model as one step of a larger job. Nothing is uploaded and
there is no cost per model.

It is an unofficial wrapper. TRELLIS is the work of Microsoft Research; its code is
MIT-licensed. This repo holds no TRELLIS code or weights: the install script fetches them.
Check the terms of the weights and of TRELLIS's dependencies yourself before shipping anything
made with them.

Its companion is [UniRigMCP](https://github.com/digitalblack210/UniRigMCP), which gives the
model a skeleton. The two share nothing but their shape, and either works alone.

## What you need

- An NVIDIA GPU. Tested on an RTX 5060 Ti with 16 GB, where a model peaks at about 10 GB; less
  has not been tried.
- Linux, or Windows with WSL 2 (tested on Ubuntu 24.04 under WSL).
- About 20 GB of disk for TRELLIS and its weights. The weights download to the Hugging Face
  cache the first time a model is made.
- Python 3.10 or newer for the server itself.

## Install

1. Get this repo and install TRELLIS. On Windows run the install inside WSL; it needs no sudo
   and is safe to rerun.

   ```bash
   git clone https://github.com/digitalblack210/TrellisMCP.git
   cd TrellisMCP
   bash scripts/install.sh
   ```

   It installs into `~/trellis`. Set `TRELLIS_HOME` first to put it somewhere else. The script
   is not Microsoft's `setup.sh`: that one installs a PyTorch too old for RTX 50-series cards.
   This one uses Python 3.11, CUDA 12.8 and PyTorch 2.7.1, and pins TRELLIS to the commit it
   was tested against.

2. Install the server's one dependency. On Windows do this in Windows, not WSL.

   ```bash
   python -m venv .venv
   .venv/Scripts/pip install -r requirements.txt     # Linux: .venv/bin/pip
   ```

3. Tell your agent about it. `TRELLIS_MCP_ROOTS` lists the folders the server may read pictures
   from and save results to, separated by `;`. It refuses every other path.

   Claude Code, in a project's `.mcp.json`:

   ```json
   {
     "mcpServers": {
       "trellis": {
         "command": "C:\\TrellisMCP\\.venv\\Scripts\\python.exe",
         "args": ["C:\\TrellisMCP\\server.py"],
         "env": { "TRELLIS_MCP_ROOTS": "C:\\MyGame\\art" }
       }
     }
   }
   ```

   Codex, in `~/.codex/config.toml`:

   ```toml
   [mcp_servers.trellis]
   command = "C:\\TrellisMCP\\.venv\\Scripts\\python.exe"
   args = ["C:\\TrellisMCP\\server.py"]
   env = { TRELLIS_MCP_ROOTS = "C:\\MyGame\\art" }
   tool_timeout_sec = 90
   ```

## The tools

| Tool | What it does |
|---|---|
| `start_model(picture, seed=1, ...)` | Starts a job and returns at once with a `job_id`. |
| `get_model(job_id, wait_seconds=45, save_to=None)` | Waits up to 50 s, then reports. Call again while `status` is `running`. |
| `list_models(limit=10)` | Recent jobs, newest first. |

A normal run is one `start_model`, then `get_model` once or twice. Jobs are started and polled,
not awaited, because a model takes longer than most clients let one tool call run.

`picture` is an absolute path to a PNG, JPEG or WebP under an allowed folder. A finished job
gives paths to the GLB, a views picture and TRELLIS's log. `save_to` copies the GLB and the
views picture into a folder of your choosing, with a small provenance file beside them.

`start_model` also takes:

| Setting | Default | What it changes |
|---|---|---|
| `seed` | 1 | The model is sampled. Another seed gives another shape. |
| `steps` | TRELLIS's own | Sampler steps, 1 to 100. |
| `texture_size` | 1024 | 512, 1024 or 2048 pixels. |
| `simplify` | 0.95 | Share of triangles removed on export. Lower it when thin parts come out broken. |
| `more_pictures` | none | Up to 3 further pictures of the same object from other sides. |

## What to expect

- **Give it one whole object, centred.** A transparent background is used as it is; otherwise
  TRELLIS removes the background itself.
- **Look at the views picture.** It shows six camera angles, textured on top and shape only
  below. An agent that can read pictures should check it before accepting a model.
- **A flat drawing can come back as a picture on a card.** The job's `result` has `flat_card`,
  true when the mesh looks like that. Another seed, a larger cut of the picture or a slightly
  tilted picture has each fixed it.
- **The side the picture does not show is guessed.** The far flank of a side-view creature
  often comes out plain.
- **The result is one fused mesh with one texture.** There are no separate parts, no rig and
  no real-world scale. Wheels do not turn and doors do not open.
- **`more_pictures` is a trick, not a trained mode.** TRELLIS was trained on one picture; it
  helps only when the pictures really are the same object.
- **One job at a time.** A second `start_model` while one is running is refused with a message
  saying so. Anything else using the GPU's memory can make a job fail.
- **A failed job reports the end of TRELLIS's log** in `error`, so the agent can see whether it
  ran out of memory, could not read the picture, or something else.

## Layout

```
server.py             the MCP server; also importable for tests
scripts/install.sh    TRELLIS install for RTX 50-series cards
scripts/generate.sh   one picture to one model; sets up TRELLIS's environment
scripts/generate.py   the TRELLIS calls, run with TRELLIS's own Python
tests/                run with: python -m pytest
jobs/                 one folder per job (ignored by git)
```

Job state lives in `jobs/<id>/job.json`, so a job keeps running and stays readable if the
server or the agent restarts. Set `TRELLIS_MCP_JOBS` to keep jobs somewhere else.

## Licence

The code in this repo is under the [MIT licence](LICENSE). That covers this wrapper only:
TRELLIS, its dependencies and its weights carry their own terms.
