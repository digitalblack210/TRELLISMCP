"""TRELLIS as MCP tools: an agent hands over a picture and gets back a textured 3D model.

Run by an MCP client (Claude Code, Codex, anything that speaks the protocol) over stdio:

    python server.py

TRELLIS itself runs on Linux with an NVIDIA GPU. On Windows this server starts it
inside WSL; on Linux it starts it directly. Either way TRELLIS must already be
installed with scripts/install.sh.

Three things keep the server in its lane:

  - It reads pictures only from folders named in TRELLIS_MCP_ROOTS, and copies
    results only into those folders. An agent cannot use it to read or write
    anywhere else on the machine.
  - One model is made at a time. The GPU is the limit, not the server.
  - Nothing is printed to stdout but the protocol. A stray print would corrupt
    the stream the client is reading.
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = Path(os.getenv("TRELLIS_MCP_JOBS", ROOT / "jobs")).resolve()
RUN_SCRIPT = ROOT / "scripts" / "generate.sh"
PICTURE_TYPES = {".png", ".jpg", ".jpeg", ".webp"}
STAGES = ("load", "generate", "views", "export", "done")
DEFAULT_SEED = 1
DEFAULT_TEXTURE = 1024
DEFAULT_SIMPLIFY = 0.95
MAX_MORE = 3    # further pictures of the same object
MAX_WAIT = 50   # seconds one call may block; MCP clients give up on a tool that takes much longer
ON_WINDOWS = os.name == "nt"


def allowed_roots():
    """The job folder, plus every folder in TRELLIS_MCP_ROOTS (separated by ; or the OS separator)."""
    raw = os.getenv("TRELLIS_MCP_ROOTS", "")
    parts = raw.replace(os.pathsep, ";").split(";")
    return [JOBS] + [Path(part.strip()).expanduser().resolve() for part in parts if part.strip()]


def inside_allowed(path):
    """The path, resolved, if it sits under an allowed folder. Otherwise ValueError."""
    resolved = Path(path).expanduser().resolve()
    roots = allowed_roots()
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError("%s is outside the folders this server may use: %s. Add its folder to TRELLIS_MCP_ROOTS."
                         % (path, ", ".join(map(str, roots))))
    return resolved


def picture_file(path):
    resolved = inside_allowed(path)
    if resolved.suffix.lower() not in PICTURE_TYPES or not resolved.is_file():
        raise ValueError("%s is not a picture that exists (%s)." % (path, ", ".join(sorted(PICTURE_TYPES))))
    return resolved


def linux_path(path):
    """How the path is spelled where TRELLIS runs: /mnt/c/... inside WSL, unchanged on Linux."""
    path = Path(path)
    if not ON_WINDOWS:
        return str(path)
    drive = path.drive.rstrip(":")
    if len(drive) != 1:
        raise ValueError("%s is not on a lettered drive, so WSL cannot reach it." % path)
    return "/mnt/%s/%s" % (drive.lower(), "/".join(path.parts[1:]))


def settings(seed=None, steps=None, texture_size=None, simplify=None):
    """The numbers a job runs with, checked. steps 0 means TRELLIS's own default."""
    seed = DEFAULT_SEED if seed is None else int(seed)
    steps = 0 if steps is None else int(steps)
    texture_size = DEFAULT_TEXTURE if texture_size is None else int(texture_size)
    simplify = DEFAULT_SIMPLIFY if simplify is None else float(simplify)
    if seed < 0:
        raise ValueError("seed must be 0 or more, not %d." % seed)
    if steps and not 1 <= steps <= 100:
        raise ValueError("steps must be between 1 and 100, not %d." % steps)
    if texture_size not in (512, 1024, 2048):
        raise ValueError("texture_size must be 512, 1024 or 2048, not %d." % texture_size)
    if not 0 <= simplify < 1:
        raise ValueError("simplify is the share of triangles removed: 0 up to, but not, 1. Not %s." % simplify)
    return {"seed": seed, "steps": steps, "texture_size": texture_size, "simplify": simplify}


def stage_from_log(text):
    """The last stage the run script announced. None before the first."""
    stage = None
    for line in text.splitlines():
        if line.startswith("=== "):
            word = line[4:].split(" ", 1)[0]
            if word in STAGES:
                stage = word
    return stage


def _state_path(job_id):
    if not job_id or any(char not in "0123456789abcdef-" for char in job_id):
        raise ValueError("%r is not a job id this server made." % job_id)
    return JOBS / job_id / "job.json"


def _read_state(job_id):
    path = _state_path(job_id)
    if not path.is_file():
        raise ValueError("There is no job %s." % job_id)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(state):
    path = _state_path(state["id"])
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _process_alive(pid):
    if ON_WINDOWS:
        result = subprocess.run(["tasklist", "/FI", "PID eq %d" % pid, "/NH", "/FO", "CSV"], capture_output=True,
                                text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        return '"%d"' % pid in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _files(state):
    folder = JOBS / state["id"]
    return {"glb": folder / (state["name"] + ".glb"), "views_png": folder / (state["name"] + "_views.png"),
            "log": folder / (state["name"] + "_trellis.log")}


def _refresh(state):
    """Bring a job's recorded status up to date with what is on disk. Saves the state if it changed."""
    if state["status"] != "running":
        return state
    files = _files(state)
    text = files["log"].read_text(encoding="utf-8", errors="replace") if files["log"].is_file() else ""
    stage = stage_from_log(text)
    if stage == "done" and files["glb"].is_file():
        # When the file was written, not when somebody next asked: a job may sit unpolled.
        state.update(status="done", stage="done", finished=files["glb"].stat().st_mtime)
    elif not _process_alive(state["pid"]):
        # The script stopped without reaching "done". Its log is the only account of why.
        tail = [line for line in text.splitlines() if line.strip()][-12:]
        state.update(status="failed", stage=stage,
                     finished=files["log"].stat().st_mtime if files["log"].is_file() else time.time(),
                     error="TRELLIS stopped during the %s stage. Last lines of its log:\n%s"
                           % (stage or "start-up", "\n".join(tail) or "(the log is empty)"))
    else:
        if stage == state.get("stage"):
            return state
        state["stage"] = stage
    _write_state(state)
    return state


def _all_jobs():
    jobs = []
    if JOBS.is_dir():
        for path in JOBS.glob("*/job.json"):
            try:
                jobs.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue   # a half-written or foreign folder is not a job
    return sorted(jobs, key=lambda job: job.get("created", 0), reverse=True)


def _describe(state):
    """What an agent needs to know about a job, with real paths once there is something at them."""
    end = state.get("finished") or time.time()
    out = {"job_id": state["id"], "status": state["status"], "stage": state.get("stage"), "name": state["name"],
           "source_picture": state["source"], "more_pictures": state.get("more", []), "settings": state["settings"],
           "elapsed_seconds": round(end - state["created"]), "error": state.get("error")}
    out["files"] = {kind: str(path) for kind, path in _files(state).items() if path.is_file()}
    if state["status"] == "done":
        result = JOBS / state["id"] / (state["name"] + "_result.json")
        try:
            out["result"] = json.loads(result.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass   # the model is there; only its measurements are not
    return out


def start_model(picture, seed=None, steps=None, texture_size=None, simplify=None, more_pictures=None):
    source = picture_file(picture)
    more = [picture_file(path) for path in (more_pictures or [])]
    if len(more) > MAX_MORE:
        raise ValueError("At most %d further pictures; %d were given." % (MAX_MORE, len(more)))
    chosen = settings(seed, steps, texture_size, simplify)
    if not RUN_SCRIPT.is_file():
        raise RuntimeError("%s is missing, so there is nothing to run." % RUN_SCRIPT)
    for job in _all_jobs():
        if _refresh(job)["status"] == "running":
            raise RuntimeError("Job %s is still making %s. One model is made at a time: wait for it, then try again."
                               % (job["id"], job["name"]))
    job_id = "%s-%s" % (time.strftime("%Y%m%d-%H%M%S"), os.urandom(3).hex())
    folder = JOBS / job_id
    folder.mkdir(parents=True)
    arguments = (["bash", linux_path(RUN_SCRIPT), linux_path(source), linux_path(folder), str(chosen["seed"]),
                  str(chosen["steps"]), str(chosen["simplify"]), str(chosen["texture_size"])]
                 + [linux_path(path) for path in more])
    if ON_WINDOWS:
        command, flags = ["wsl.exe", "-e"] + arguments, subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        command, flags = arguments, 0
    try:
        # Detached from this server's pipes: stdout belongs to the protocol, and the job must outlive a restart.
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   creationflags=flags, start_new_session=not ON_WINDOWS)
    except OSError as exc:
        shutil.rmtree(folder, ignore_errors=True)
        raise RuntimeError("Could not start TRELLIS (%s). On Windows this needs WSL; on Linux it needs bash." % exc) from exc
    state = {"id": job_id, "status": "running", "stage": None, "name": "%s_s%d" % (source.stem, chosen["seed"]),
             "source": str(source), "more": [str(path) for path in more], "settings": chosen, "pid": process.pid,
             "created": time.time()}
    _write_state(state)
    return _describe(state)


def get_model(job_id, wait_seconds=45, save_to=None):
    deadline = time.time() + max(0, min(int(wait_seconds), MAX_WAIT))
    while True:
        state = _refresh(_read_state(job_id))
        if state["status"] != "running" or time.time() >= deadline:
            break
        time.sleep(2)
    out = _describe(state)
    if save_to and state["status"] == "done":
        folder = inside_allowed(save_to)
        folder.mkdir(parents=True, exist_ok=True)
        saved = {}
        for kind, suffix in (("glb", ".glb"), ("views_png", "_views.png")):
            if kind in out["files"]:
                target = folder / (state["name"] + suffix)
                shutil.copyfile(out["files"][kind], target)
                saved[kind] = str(target)
        record = folder / (state["name"] + ".provenance.json")
        record.write_text(json.dumps({"generator": "microsoft/TRELLIS-image-large", "source_picture": state["source"],
                                      "more_pictures": state.get("more", []), "settings": state["settings"],
                                      "result": out.get("result")}, indent=2), encoding="utf-8")
        saved["provenance_json"] = str(record)
        out["saved"] = saved
    return out


def list_models(limit=10):
    return [_describe(_refresh(job)) for job in _all_jobs()[:max(1, min(int(limit), 20))]]


def build_server():
    # Imported here so the functions above can be used, and tested, without the MCP SDK.
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    def told(call):
        # The SDK hides the text of any other exception behind "Error executing
        # tool", which leaves the agent with nothing to act on. These are all
        # reasons it can act on: a bad path, a busy GPU, a missing install.
        try:
            return call()
        except (ValueError, RuntimeError) as exc:
            raise ToolError(str(exc)) from exc

    server = MCPServer(
        "trellis",
        instructions=(
            "Turns one picture of an object into a textured 3D model (GLB) on this machine's GPU, using "
            "Microsoft's TRELLIS. Call start_model, then get_model with the job id until status is 'done' or "
            "'failed'. One model is made at a time; a second start while one is running is refused, so wait and "
            "retry. A model takes about a minute. Look at the views picture before accepting a model: the result "
            "is sampled, and another seed can fix a bad one."
        ),
    )

    @server.tool(name="start_model")
    def start(picture: str, seed: int | None = None, steps: int | None = None, texture_size: int | None = None,
              simplify: float | None = None, more_pictures: list[str] | None = None) -> dict:
        """Start making a 3D model from one picture.

        picture: absolute path on this machine to a PNG, JPEG or WebP under a folder named in TRELLIS_MCP_ROOTS.
            One whole object, centred, works best. A transparent background is used as it is; otherwise the
            background is removed automatically.
        seed: leave it out for the default, 1. The model is sampled, so another seed gives another shape. Try
            one when the result is wrong or 'flat_card' is true.
        steps: sampler steps, 1 to 100. Leave it out for TRELLIS's own default.
        texture_size: 512, 1024 (default) or 2048 pixels.
        simplify: share of triangles removed on export, default 0.95. Lower it (0.9, 0.8) when thin parts come
            out broken or as loose pieces; the file gets larger.
        more_pictures: up to 3 further pictures of the SAME object from other sides. TRELLIS was not trained
            for this, so it helps only when they really are the same object.

        Returns the job, including its job_id. The model is not ready yet: call get_model with that id.
        """
        return told(lambda: start_model(picture, seed, steps, texture_size, simplify, more_pictures))

    @server.tool(name="get_model")
    def get(job_id: str, wait_seconds: int = 45, save_to: str | None = None) -> dict:
        """Check on a job, waiting up to wait_seconds (50 at most) for it to finish.

        While status is 'running', 'stage' says where it is ('load', 'generate', 'views' or 'export'); call
        again. When status is 'done', 'files' holds absolute paths to the GLB, the views picture and the log.
        The views picture shows six camera angles, textured on top and shape only below: look at it. 'result'
        holds timings, peak GPU memory, the model's size and 'flat_card', which is true when the mesh looks
        like a picture on a flat sheet instead of a solid object. When status is 'failed', 'error' holds the
        end of TRELLIS's log.

        save_to: optional folder under an allowed root. When the job is done, the GLB and the views picture
            are copied there with a small provenance file beside them, and 'saved' gives their paths.
        """
        return told(lambda: get_model(job_id, wait_seconds, save_to))

    @server.tool(name="list_models")
    def recent(limit: int = 10) -> list[dict]:
        """The most recent jobs, newest first, with their files when finished."""
        return told(lambda: list_models(limit))

    return server


if __name__ == "__main__":
    try:
        build_server().run("stdio")
    except Exception as exc:   # anything said here must go to stderr, never stdout
        print("trellis MCP server could not start: %s" % exc, file=sys.stderr)
        raise
