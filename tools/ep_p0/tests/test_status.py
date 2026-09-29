"""status.json 初始化与读写的单元测试（任务 2.1）。"""

from __future__ import annotations

import json

import pytest

from tools.ep_p0 import status

SECTIONS = {
    "repo", "dependency_declarations", "ci", "ignore_checks", "auth_patterns",
    "identity_checks", "credential_scans", "pushes", "manual", "waits", "gates",
    "p1_decision", "master_spec_suggestions", "handover",
}


def test_init_builds_empty_structure(tmp_path):
    records = tmp_path / "records" / "ep-p0"
    path, created = status.init(records, today="2026-09-29", secrets=[])
    assert created and path.is_file()

    data = status.load(records)
    assert SECTIONS <= set(data)
    gates = {g["id"]: g for g in data["gates"]}
    assert [g["id"] for g in data["gates"]] == [f"G{i}" for i in range(1, 10)] + ["S1", "S2", "S3"]
    assert all(gates[f"G{i}"]["status"] == "unmet" for i in range(1, 10))
    assert all(gates[s]["status"] is None and gates[s]["updated"] is None for s in ("S1", "S2", "S3"))
    assert status.gate_status_label(gates["S1"]["status"]) == "未记录"

    manual = data["manual"]
    assert set(manual) == {"mentor_record", "public_email", "cim", "pr_strategy"}
    assert manual["cim"]["status"] == "not_applied"
    assert manual["pr_strategy"]["confirmation"] == "pending"
    assert manual["pr_strategy"]["description"] == status.DEFAULT_PR_STRATEGY_TEXT
    assert data["p1_decision"] == {"value": "pending", "date": None}


def test_init_does_not_overwrite(tmp_path):
    records = tmp_path / "rec"
    status.init(records, secrets=[])
    data = status.load(records)
    data["waits"].append({"item": "x", "party": "mentor", "requested_date": "2026-09-29", "channel": "mail"})
    status.save(data, records, secrets=[])
    before = status.status_path(records).read_bytes()

    _, created = status.init(records, secrets=[])
    assert not created
    assert status.status_path(records).read_bytes() == before


def test_main_exit_codes(tmp_path, capsys, monkeypatch):
    for name in ("KAIWU_USER_ID", "KAIWU_SDK_CODE"):
        monkeypatch.delenv(name, raising=False)
    records = tmp_path / "rec"
    assert status.main(["--records", str(records)]) == 0
    assert status.main(["--records", str(records)]) == 0
    assert "未覆盖" in capsys.readouterr().out


def test_save_rejects_credentials_without_writing(tmp_path, monkeypatch):
    secret = "s3cr3t-VALUE-xyz"
    monkeypatch.setenv("KAIWU_SDK_CODE", secret)
    records = tmp_path / "rec"
    data = status.empty_status("2026-09-29")
    data["waits"].append({"item": f"token {secret}", "party": "user"})

    with pytest.raises(status.CredentialLeakError) as info:
        status.save(data, records)
    assert secret not in str(info.value)
    assert not status.status_path(records).exists()


def test_main_returns_1_on_credential_hit(tmp_path, monkeypatch):
    monkeypatch.delenv("KAIWU_USER_ID", raising=False)
    monkeypatch.setenv("KAIWU_SDK_CODE", "Dep_PR")  # 默认 PR_Strategy 文本含该值
    assert status.main(["--records", str(tmp_path / "rec")]) == 1
    assert not (tmp_path / "rec" / "status.json").exists()


def test_save_relativizes_workspace_paths_and_rejects_others(tmp_path):
    ws = tmp_path / "ws"
    records = ws / "records" / "ep-p0"
    data = status.empty_status("2026-09-29")
    data["repo"]["clone_path"] = str(ws / "upstream" / "kaiwu-pytorch-plugin")
    status.save(data, records, workspace_root=ws, secrets=[])
    assert status.load(records)["repo"]["clone_path"] == "upstream/kaiwu-pytorch-plugin"

    data["repo"]["clone_path"] = "Z:\\elsewhere\\clone"
    with pytest.raises(status.AbsolutePathError):
        status.save(data, records, workspace_root=ws, secrets=[])
    data["repo"]["clone_path"] = None
    data["waits"].append({"item": f"see {ws}\\x.log", "party": "user"})
    with pytest.raises(status.AbsolutePathError):
        status.save(data, records, workspace_root=ws, secrets=[])


def test_save_is_deterministic_utf8_and_atomic(tmp_path):
    records = tmp_path / "rec"
    data = status.empty_status("2026-09-29")
    path = status.save(data, records, secrets=[])
    first = path.read_bytes()
    reordered = dict(reversed(list(data.items())))
    status.save(reordered, records, secrets=[])
    assert path.read_bytes() == first
    assert first.endswith(b"\n") and b"\r\n" not in first
    text = first.decode("utf-8")
    assert "能量/QUBO" in text  # 中文原样保存（ensure_ascii=False）
    assert json.loads(text) == data
    assert [p.name for p in records.iterdir()] == ["status.json"]  # 无残留临时文件
