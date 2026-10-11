import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server


@pytest.fixture
def roots(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    monkeypatch.setattr(server, "JOBS", tmp_path / "jobs")
    monkeypatch.setenv("TRELLIS_MCP_ROOTS", str(allowed))
    return allowed


def running_job(name="rat_s1"):
    folder = server.JOBS / "20260101-000000-abcdef"
    folder.mkdir(parents=True)
    state = {"id": folder.name, "status": "running", "stage": None, "name": name, "source": "x", "more": [],
             "settings": server.settings(), "pid": 1, "created": 0}
    server._write_state(state)
    return folder


def test_stage_is_the_last_one_announced():
    assert server.stage_from_log("") is None
    assert server.stage_from_log("=== load 10:00:01\nnoise\n=== generate 10:00:40\n") == "generate"
    assert server.stage_from_log("=== load 10:00:01\n=== nonsense 1\n") == "load"


def test_picture_outside_the_roots_is_refused(roots, tmp_path):
    outside = tmp_path / "elsewhere.png"
    outside.write_bytes(b"png")
    with pytest.raises(ValueError, match="outside the folders"):
        server.picture_file(outside)


def test_dot_dot_cannot_escape_the_roots(roots, tmp_path):
    (tmp_path / "elsewhere.png").write_bytes(b"png")
    with pytest.raises(ValueError, match="outside the folders"):
        server.picture_file(roots / ".." / "elsewhere.png")


def test_picture_inside_the_roots_is_accepted(roots):
    inside = roots / "wolf.png"
    inside.write_bytes(b"png")
    assert server.picture_file(inside) == inside.resolve()


def test_wrong_file_type_is_refused(roots):
    model = roots / "wolf.glb"
    model.write_bytes(b"glTF")
    with pytest.raises(ValueError, match="not a picture"):
        server.picture_file(model)


def test_settings_have_defaults():
    assert server.settings() == {"seed": 1, "steps": 0, "texture_size": 1024, "simplify": 0.95}


@pytest.mark.parametrize("bad", [{"seed": -1}, {"steps": 500}, {"texture_size": 999}, {"simplify": 1}, {"simplify": -0.1}])
def test_settings_out_of_range_are_refused(bad):
    with pytest.raises(ValueError):
        server.settings(**bad)


def test_too_many_further_pictures_are_refused(roots):
    pictures = []
    for index in range(5):
        picture = roots / ("side%d.png" % index)
        picture.write_bytes(b"png")
        pictures.append(str(picture))
    with pytest.raises(ValueError, match="At most"):
        server.start_model(pictures[0], more_pictures=pictures[1:])


def test_job_id_cannot_name_another_folder():
    with pytest.raises(ValueError, match="not a job id"):
        server.get_model("../../somewhere")


def test_unknown_job_is_reported(roots):
    with pytest.raises(ValueError, match="no job"):
        server.get_model("20260101-000000-abcdef", wait_seconds=0)


@pytest.mark.skipif(not server.ON_WINDOWS, reason="WSL path spelling only applies on Windows")
def test_windows_path_becomes_a_wsl_path():
    assert server.linux_path(Path(r"C:\Art\My Rat\rat.png")) == "/mnt/c/Art/My Rat/rat.png"


def test_a_stopped_job_fails_with_the_end_of_its_log(roots, monkeypatch):
    folder = running_job()
    (folder / "rat_s1_trellis.log").write_text("=== load 10:00:01\n=== generate 10:00:20\nCUDA out of memory\n")
    monkeypatch.setattr(server, "_process_alive", lambda pid: False)
    out = server.get_model(folder.name, wait_seconds=0)
    assert out["status"] == "failed" and out["stage"] == "generate"
    assert "CUDA out of memory" in out["error"]


def test_a_finished_job_reports_and_saves_its_files(roots, monkeypatch):
    folder = running_job()
    (folder / "rat_s1_trellis.log").write_text("=== load 1\n=== generate 2\n=== views 3\n=== export 4\n=== done 5\n")
    (folder / "rat_s1.glb").write_bytes(b"glTF")
    (folder / "rat_s1_views.png").write_bytes(b"png")
    (folder / "rat_s1_result.json").write_text(json.dumps({"flat_card": False, "peak_gpu_memory_gb": 9.7}))
    monkeypatch.setattr(server, "_process_alive", lambda pid: False)
    out = server.get_model(folder.name, wait_seconds=0, save_to=str(roots / "models"))
    assert out["status"] == "done" and out["result"]["flat_card"] is False
    assert set(out["files"]) == {"glb", "views_png", "log"}
    assert (roots / "models" / "rat_s1.glb").read_bytes() == b"glTF"
    record = json.loads((roots / "models" / "rat_s1.provenance.json").read_text())
    assert record["generator"] == "microsoft/TRELLIS-image-large" and record["settings"]["seed"] == 1


def test_results_cannot_be_saved_outside_the_roots(roots, tmp_path, monkeypatch):
    folder = running_job()
    (folder / "rat_s1_trellis.log").write_text("=== done 5\n")
    (folder / "rat_s1.glb").write_bytes(b"glTF")
    with pytest.raises(ValueError, match="outside the folders"):
        server.get_model(folder.name, wait_seconds=0, save_to=str(tmp_path / "elsewhere"))


def test_a_second_job_is_refused_while_one_runs(roots, monkeypatch):
    running_job()
    monkeypatch.setattr(server, "_process_alive", lambda pid: True)
    picture = roots / "wolf.png"
    picture.write_bytes(b"png")
    with pytest.raises(RuntimeError, match="still making"):
        server.start_model(str(picture))


def test_the_install_folder_is_handed_to_the_script(monkeypatch):
    monkeypatch.delenv("TRELLIS_HOME", raising=False)
    plain = server.run_command(["run.sh", "a"])
    assert plain[-3:] == ["bash", "run.sh", "a"] and not any("TRELLIS_HOME" in part for part in plain)
    monkeypatch.setenv("TRELLIS_HOME", "/opt/tools")
    moved = server.run_command(["run.sh", "a"])
    assert moved[-5:] == ["env", "TRELLIS_HOME=/opt/tools", "bash", "run.sh", "a"]
    assert (moved[:2] == ["wsl.exe", "-e"]) == server.ON_WINDOWS
