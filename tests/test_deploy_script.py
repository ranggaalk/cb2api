import hashlib
import os
import shutil
import stat
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def copy_deploy_fixture(tmp_path):
    for name in ("deploy.sh", ".env.example"):
        shutil.copy2(PROJECT_ROOT / name, tmp_path / name)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shutil.copy2(
        PROJECT_ROOT / "config" / "codebuddy_api_keys.example.txt",
        config_dir / "codebuddy_api_keys.example.txt",
    )
    return tmp_path / "deploy.sh"


def run_bootstrap(script, *args):
    env = os.environ.copy()
    env["DEPLOY_BOOTSTRAP_ONLY"] = "1"
    return subprocess.run(
        ["bash", str(script), *args],
        cwd=script.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def env_values(path):
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.lstrip().startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value
    return values


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bootstrap_creates_secure_passthrough_environment(tmp_path):
    script = copy_deploy_fixture(tmp_path)
    result = run_bootstrap(script)

    assert result.returncode == 0, result.stderr
    env_file = tmp_path / ".env"
    values = env_values(env_file)
    assert values["CODEBUDDY_CLIENT_AUTH_MODE"] == "passthrough"
    assert values["CODEBUDDY_UPSTREAM_API_KEY_HEADER"] == "bearer"
    assert values["CODEBUDDY_HOST"] == "0.0.0.0"
    assert values["CODEBUDDY_PORT"] == "8001"
    assert len(values["CODEBUDDY_PASSWORD"]) == 64
    assert len(values["CODEBUDDY_ADMIN_PASSWORD"]) == 64
    assert values["CODEBUDDY_PASSWORD"] != values["CODEBUDDY_ADMIN_PASSWORD"]
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    assert (tmp_path / "config" / "codebuddy_api_keys.txt").exists()
    assert (tmp_path / ".codebuddy_creds").is_dir()


def test_bootstrap_is_idempotent_and_preserves_existing_files(tmp_path):
    script = copy_deploy_fixture(tmp_path)
    first = run_bootstrap(
        script,
        "--relay-password",
        "fixed-relay",
        "--admin-password",
        "fixed-admin",
    )
    assert first.returncode == 0, first.stderr

    env_file = tmp_path / ".env"
    key_file = tmp_path / "config" / "codebuddy_api_keys.txt"
    key_file.write_text("real-key-must-survive\n", encoding="utf-8")
    env_digest = digest(env_file)
    key_digest = digest(key_file)

    second = run_bootstrap(script)
    assert second.returncode == 0, second.stderr
    assert digest(env_file) == env_digest
    assert digest(key_file) == key_digest
    values = env_values(env_file)
    assert values["CODEBUDDY_PASSWORD"] == "fixed-relay"
    assert values["CODEBUDDY_ADMIN_PASSWORD"] == "fixed-admin"


def test_bootstrap_supports_mode_port_and_header_overrides(tmp_path):
    script = copy_deploy_fixture(tmp_path)
    result = run_bootstrap(
        script,
        "--client-auth-mode",
        "hybrid",
        "--upstream-header",
        "both",
        "--port",
        "18001",
    )

    assert result.returncode == 0, result.stderr
    values = env_values(tmp_path / ".env")
    assert values["CODEBUDDY_CLIENT_AUTH_MODE"] == "hybrid"
    assert values["CODEBUDDY_UPSTREAM_API_KEY_HEADER"] == "both"
    assert values["CODEBUDDY_PORT"] == "18001"


def test_invalid_options_fail_before_writing_environment(tmp_path):
    script = copy_deploy_fixture(tmp_path)
    result = run_bootstrap(script, "--client-auth-mode", "invalid")

    assert result.returncode != 0
    assert not (tmp_path / ".env").exists()
