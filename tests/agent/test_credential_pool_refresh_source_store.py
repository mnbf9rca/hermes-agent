"""A rotating OAuth grant keeps one authority across profile refreshes.

This file collects 150 cases. With all five production files restored to
upstream, the pinned run has 29 passes, 117 failures and four provider-specific
skips. Passing invariants protect existing behavior or guard against regressions
introduced by this change; they are not claimed as reproductions. The complete
set of passing case IDs is:

    test_borrowed_load_save_failure_keeps_usable_pool[openai-codex-False]
    test_borrowed_load_save_failure_keeps_usable_pool[xai-oauth-False]
    test_borrowed_pool_add_switches_new_grant_to_active_store[openai-codex]
    test_borrowed_pool_add_switches_new_grant_to_active_store[xai-oauth]
    test_classic_three_references_share_one_rotation[openai-codex-True]
    test_device_row_without_singleton_adopts_peer_rotation[xai-oauth]
    test_expired_peer_successor_is_refreshed[openai-codex]
    test_expired_peer_successor_is_refreshed[xai-oauth]
    test_independent_manual_grant_never_adopts_or_overwrites_root[xai-oauth]
    test_loaded_root_pool_keeps_authority_when_active_context_changes[openai-codex]
    test_loaded_root_pool_keeps_authority_when_active_context_changes[xai-oauth]
    test_new_login_stays_in_active_profile[openai-codex]
    test_other_manual_oauth_source_keeps_pool_authority[xai-oauth]
    test_owned_row_never_seeds_or_refreshes_foreign_singleton[openai-codex-True-device_code]
    test_owned_row_never_seeds_or_refreshes_foreign_singleton[openai-codex-True-manual:device_code]
    test_owned_row_never_seeds_or_refreshes_foreign_singleton[xai-oauth-False-manual:device_code]
    test_owned_row_never_seeds_or_refreshes_foreign_singleton[xai-oauth-True-device_code]
    test_owned_row_never_seeds_or_refreshes_foreign_singleton[xai-oauth-True-manual:device_code]
    test_peer_refresh_without_access_token_still_refreshes[openai-codex-device_code]
    test_peer_refresh_without_access_token_still_refreshes[openai-codex-manual:device_code]
    test_refresh_locks_and_writes_source_store[openai-codex-True-device_code]
    test_refresh_locks_and_writes_source_store[xai-oauth-True-device_code]
    test_rejection_adopts_successor_with_shared_previous_access_token[openai-codex-False-singleton]
    test_root_peer_adopts_profile_rotation[xai-oauth]
    test_same_store_independent_grant_is_not_singleton_alias[xai-oauth]
    test_singleton_reseed_reaches_its_existing_row_on_disk[openai-codex]
    test_singleton_reseed_reaches_its_existing_row_on_disk[xai-oauth]
    test_terminal_grant_is_not_reported_or_returned_by_runtime[openai-codex-device_code]
    test_terminal_grant_is_not_reported_or_returned_by_runtime[xai-oauth-device_code]
"""

import json
import os
import subprocess
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from pathlib import Path

import pytest

from agent import credential_pool as CP
from hermes_cli import auth as A


@pytest.fixture(params=["openai-codex", "xai-oauth"])
def grant_stores(tmp_path, monkeypatch, request):
    repo = Path(__file__).resolve().parents[2]
    assert Path(CP.__file__).resolve().is_relative_to(repo)
    assert Path(A.__file__).resolve().is_relative_to(repo)
    provider = request.param
    root = tmp_path / "root" / "auth.json"
    profiles = [tmp_path / "profiles" / name / "auth.json" for name in ("a", "b", "c")]
    active = ContextVar("test_auth_path", default=profiles[0])
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(profiles[0].parent))
    monkeypatch.setattr(A, "_auth_file_path", active.get)
    monkeypatch.setattr(A, "_global_auth_file_path", lambda: None if active.get() == root else root)
    monkeypatch.setattr(CP, "_global_auth_file_path", lambda: None if active.get() == root else root)
    monkeypatch.setattr(CP, "_load_config_safe", lambda: {})
    for path in [root, *profiles]:
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"providers": {}}), encoding="utf-8")

    def seed(path, source="manual:device_code", token="original", singleton=True):
        entry = CP.PooledCredential(
            provider=provider, id=token, label="test grant", auth_type="oauth",
            priority=0, source=source, access_token="access-" + token,
            refresh_token=token,
        )
        store = {"providers": {}, "credential_pool": {provider: [entry.to_dict()]}}
        if singleton:
            store["providers"][provider] = {"tokens": {
                "access_token": entry.access_token, "refresh_token": token,
            }}
        path.write_text(json.dumps(store), encoding="utf-8")
        A._global_auth_store_cache = None
        return entry

    def pool(path):
        active.set(path)
        monkeypatch.setenv("HERMES_HOME", str(path.parent))
        entries = [CP.PooledCredential.from_dict(provider, row) for row in A.read_credential_pool(provider)]
        client = CP.CredentialPool(provider, entries)
        client._auth_store_path = path if (read(path).get("credential_pool") or {}).get(provider) else root
        return client

    return provider, root, profiles, active, seed, pool


class RotatingEndpoint:
    """Reject a consumed fake token with the provider's terminal error shape."""

    def __init__(self, provider):
        self.provider = provider
        self.consumed = set()
        self.calls = []
        self.lock = threading.Lock()
        self.before_post = None

    def __call__(self, access_token, refresh_token, **kwargs):
        if self.before_post:
            self.before_post(refresh_token)
        with self.lock:
            self.calls.append(refresh_token)
            if refresh_token in self.consumed:
                raise A.AuthError(
                    "refresh_token_reused", provider=self.provider,
                    code="refresh_token_reused" if self.provider == "openai-codex" else "xai_refresh_failed",
                    relogin_required=True,
                )
            self.consumed.add(refresh_token)
            return {"access_token": "access-" + refresh_token + "-next",
                    "refresh_token": refresh_token + "-next", "last_refresh": "2026-09-09T00:00:00Z"}


def install_endpoint(monkeypatch, provider):
    endpoint = RotatingEndpoint(provider)
    name = "refresh_codex_oauth_pure" if provider == "openai-codex" else "refresh_xai_oauth_pure"
    monkeypatch.setattr(A, name, endpoint)
    return endpoint


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("source", ["device_code", "manual:device_code"])
@pytest.mark.parametrize("root_home", [False, True])
def test_refresh_locks_and_writes_source_store(grant_stores, monkeypatch, source, root_home):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, source)
    client = pool(root if root_home else profiles[0])
    endpoint = install_endpoint(monkeypatch, provider)
    held = []
    endpoint.before_post = lambda _: held.append(getattr(A._auth_lock_holder_for(root), "depth", 0) > 0)

    result = client._refresh_entry(client.entries()[0], force=True)

    assert result is not None
    assert held == [True], "refresh must hold the source store's lock across the POST"
    assert root.with_suffix(".lock").exists()
    assert read(root)["providers"][provider]["tokens"]["refresh_token"] == result.refresh_token
    assert read(root)["credential_pool"][provider][0]["refresh_token"] == result.refresh_token
    assert provider not in read(profiles[0])["providers"]


@pytest.mark.parametrize("singleton", [False, True])
@pytest.mark.parametrize("expired_peer", [False, True])
def test_rejected_stale_token_adopts_peer_and_retries_once(grant_stores, monkeypatch, singleton, expired_peer):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=singleton)
    peer = pool(profiles[0])
    stale = pool(profiles[1])
    endpoint = install_endpoint(monkeypatch, provider)

    def peer_rotates_during_post(token):
        endpoint.before_post = None
        active.set(profiles[0])
        assert peer._refresh_entry(peer.entries()[0], force=True) is not None
        if expired_peer:
            state = read(root)
            state["credential_pool"][provider][0]["access_token"] = expired_token()
            if singleton:
                state["providers"][provider]["tokens"]["access_token"] = expired_token()
            root.write_text(json.dumps(state), encoding="utf-8")
        active.set(profiles[1])

    endpoint.before_post = peer_rotates_during_post
    result = stale._refresh_entry(stale.entries()[0], force=True)

    assert result is not None
    assert endpoint.calls == (["original", "original", "original-next"] if expired_peer else ["original", "original"])
    assert result.refresh_token == ("original-next-next" if expired_peer else "original-next")
    store = read(root)
    assert store["credential_pool"][provider][0]["refresh_token"] == result.refresh_token
    assert "last_auth_error" not in store["providers"].get(provider, {})


@pytest.mark.parametrize("singleton", [False, True])
@pytest.mark.parametrize("root_home", [False, True])
def test_dead_token_clears_only_source_store(grant_stores, monkeypatch, singleton, root_home):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=singleton)
    client = pool(root if root_home else profiles[1])
    endpoint = install_endpoint(monkeypatch, provider)
    endpoint.consumed.add("original")

    assert client._refresh_entry(client.entries()[0], force=True) is None

    if singleton:
        state = read(root)["providers"][provider]
        assert not state["tokens"].get("refresh_token")
        assert not state["tokens"].get("access_token")
        assert state["last_auth_error"]["relogin_required"] is True
    assert provider not in read(profiles[1])["providers"]
    rows = read(root)["credential_pool"].get(provider, [])
    assert rows and all(row.get("last_status") == CP.STATUS_DEAD and row.get("last_status_at") for row in rows)


def test_independent_manual_grant_never_adopts_or_overwrites_root(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    seed(profiles[2], token="independent", singleton=False)
    before = root.read_bytes()
    client = pool(profiles[2])
    endpoint = install_endpoint(monkeypatch, provider)

    result = client._refresh_entry(client.entries()[0], force=True)

    assert result is not None
    assert endpoint.calls == ["independent"]
    assert root.read_bytes() == before
    assert read(profiles[2])["credential_pool"][provider][0]["refresh_token"] == result.refresh_token
    assert provider not in read(profiles[2])["providers"]


@pytest.mark.parametrize("singleton", [True, False])
def test_concurrent_profiles_consume_original_token_once(grant_stores, monkeypatch, singleton):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=singleton)
    clients = [pool(path) for path in profiles[:2]]
    endpoint = install_endpoint(monkeypatch, provider)
    start = threading.Barrier(2)

    def refresh(index):
        active.set(profiles[index])
        start.wait(timeout=5)
        return clients[index]._refresh_entry(clients[index].entries()[0], force=True)

    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(workers.map(refresh, range(2)))

    assert all(result is not None for result in results)
    assert endpoint.calls == ["original"]
    assert results[0].refresh_token == results[1].refresh_token
    assert read(root)["credential_pool"][provider][0]["refresh_token"] == results[0].refresh_token
    assert all(provider not in read(path)["providers"] for path in profiles[:2])


@pytest.mark.parametrize("source", ["device_code", "manual:device_code"])
def test_peer_refresh_without_access_token_still_refreshes(grant_stores, monkeypatch, source):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, source=source)
    client = pool(profiles[0])
    state = read(root)
    state["providers"][provider]["tokens"] = {"refresh_token": "peer"}
    if source.startswith("manual:"):
        state["credential_pool"][provider][0].update(access_token="", refresh_token="peer")
    root.write_text(json.dumps(state), encoding="utf-8")
    endpoint = install_endpoint(monkeypatch, provider)

    result = client._refresh_entry(client.entries()[0], force=True)

    assert result is not None
    assert endpoint.calls == ["peer"]
    assert result.access_token == "access-peer-next"


def test_other_manual_oauth_source_keeps_pool_authority(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, source="manual:xai_pkce", singleton=False)
    clients = [pool(path) for path in profiles[:2]]
    endpoint = install_endpoint(monkeypatch, provider)
    first = clients[0]._refresh_entry(clients[0].entries()[0], force=True)
    second = clients[1]._refresh_entry(clients[1].entries()[0], force=True)

    assert first is not None and second is not None
    assert endpoint.calls == ["original"]
    assert second.refresh_token == first.refresh_token


def test_mixed_local_pool_and_root_singleton_preserves_lock_order(grant_stores, monkeypatch):
    from contextlib import contextmanager

    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, source="device_code")
    seed(profiles[0], source="device_code", singleton=False)
    client = pool(profiles[0])
    endpoint = install_endpoint(monkeypatch, provider)
    real_lock = A._auth_store_lock
    root_acquisitions = []

    @contextmanager
    def check_lock_order(*args, **kwargs):
        if kwargs.get("target_path") == root:
            root_acquisitions.append(getattr(A._auth_lock_holder_for(profiles[0]), "depth", 0) > 0)
        with real_lock(*args, **kwargs):
            yield

    monkeypatch.setattr(A, "_auth_store_lock", check_lock_order)
    monkeypatch.setattr(CP, "_auth_store_lock", check_lock_order)

    assert client._refresh_entry(client.entries()[0], force=True) is not None
    # A local row no longer consults or locks an unrelated root singleton.
    assert root_acquisitions == []
    assert endpoint.calls == ["original"]


def test_repeated_peer_rotations_retry_at_most_once(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    client = pool(profiles[0])
    endpoint = install_endpoint(monkeypatch, provider)

    def rotate_then_reject(token):
        state = read(root)
        row = state["credential_pool"][provider][0]
        row.update(access_token=expired_token(), refresh_token=token + "-next")
        state["providers"][provider]["tokens"] = dict(row)
        root.write_text(json.dumps(state), encoding="utf-8")
        endpoint.consumed.add(token)
        assert len(endpoint.calls) < 3, "unbounded retry"

    endpoint.before_post = rotate_then_reject
    result = client._refresh_entry(client.entries()[0], force=True)

    assert result is not None
    assert endpoint.calls == ["original", "original-next"]
    assert result.refresh_token == "original-next-next"
    assert "last_auth_error" not in read(root)["providers"][provider]


def test_expired_peer_successor_is_refreshed(grant_stores, monkeypatch):
    import base64

    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, source="device_code")
    client = pool(profiles[0])
    state = read(root)
    expired = base64.urlsafe_b64encode(b'{"exp": 1}').decode().rstrip("=")
    state["providers"][provider]["tokens"] = {"access_token": "e30." + expired + ".signature", "refresh_token": "peer"}
    root.write_text(json.dumps(state), encoding="utf-8")
    endpoint = install_endpoint(monkeypatch, provider)

    result = client._refresh_entry(client.entries()[0], force=False)

    assert result is not None
    assert endpoint.calls == ["peer"]
    assert result.refresh_token == "peer-next"


def expired_token():
    import base64
    claims = base64.urlsafe_b64encode(b'{"exp": 1}').decode().rstrip("=")
    return "e30." + claims + ".signature"


def test_poisoned_profile_shadow_cannot_own_borrowed_row(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, source="device_code")
    poison = {"providers": {provider: {"tokens": {}, "last_auth_error": {"relogin_required": True}}}}
    profiles[0].write_text(json.dumps(poison), encoding="utf-8")
    client = pool(profiles[0])
    endpoint = install_endpoint(monkeypatch, provider)
    held = []
    endpoint.before_post = lambda _: held.append(getattr(A._auth_lock_holder_for(root), "depth", 0) > 0)
    result = client._refresh_entry(client.entries()[0], force=True)
    assert result is not None and held == [True]
    assert read(root)["providers"][provider]["tokens"]["refresh_token"] == result.refresh_token
    assert read(profiles[0]) == poison


def test_terminal_failure_preserves_healthy_sibling(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=False)
    state = read(root)
    healthy = dict(state["credential_pool"][provider][0], id="healthy", source="device_code",
                   access_token="healthy-access", refresh_token="healthy-refresh")
    state["credential_pool"][provider].append(healthy)
    root.write_text(json.dumps(state), encoding="utf-8")
    client = pool(root)
    endpoint = install_endpoint(monkeypatch, provider)
    endpoint.consumed.add("original")
    client._current_id = "original"
    assert client._refresh_entry(client.entries()[0], force=True) is None
    assert client.current() is None
    assert client.peek().id == "healthy"
    rows = read(root)["credential_pool"][provider]
    assert next(row for row in rows if row["id"] == "healthy") == healthy
    failed = next(row for row in rows if row["id"] == "original")
    assert failed["last_status"] == CP.STATUS_DEAD and failed["last_status_at"]


@pytest.mark.parametrize("root_home", [False, True])
def test_stale_snapshot_preserves_other_accounts_rotation(grant_stores, monkeypatch, root_home):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=False)
    state = read(root)
    other = dict(state["credential_pool"][provider][0], id="other", access_token="other-access", refresh_token="other")
    state["credential_pool"][provider].append(other)
    root.write_text(json.dumps(state), encoding="utf-8")
    paths = [profiles[0], root if root_home else profiles[1]]
    clients = [pool(path) for path in paths]
    install_endpoint(monkeypatch, provider)
    active.set(profiles[0])
    first = clients[0]._refresh_entry(clients[0].entries()[0], force=True)
    active.set(paths[1])
    second = clients[1]._refresh_entry(clients[1].entries()[1], force=True)
    assert first is not None and second is not None
    rows = {row["id"]: row for row in read(root)["credential_pool"][provider]}
    assert rows["original"]["refresh_token"] == first.refresh_token
    assert rows["other"]["refresh_token"] == second.refresh_token


def test_rotation_commits_singleton_and_row_together(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    client = pool(profiles[0])
    install_endpoint(monkeypatch, provider)
    real_save = A._save_auth_store
    consistent = []

    def save(store, target_path=None):
        if target_path == root:
            consistent.append(store["providers"][provider]["tokens"]["refresh_token"] ==
                              store["credential_pool"][provider][0]["refresh_token"])
        return real_save(store, target_path=target_path)

    monkeypatch.setattr(A, "_save_auth_store", save)
    monkeypatch.setattr(CP, "_save_auth_store", save)
    assert client._refresh_entry(client.entries()[0], force=True) is not None
    assert consistent and all(consistent)


def test_root_peer_adopts_profile_rotation(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    root_client = pool(root)
    profile_client = pool(profiles[0])
    endpoint = install_endpoint(monkeypatch, provider)
    first = profile_client._refresh_entry(profile_client.entries()[0], force=True)
    active.set(root)
    second = root_client._refresh_entry(root_client.entries()[0], force=True)
    assert first is not None and second is not None
    assert second.refresh_token == first.refresh_token
    assert endpoint.calls == ["original"]


def test_non_pool_refresh_uses_root_authority(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    active.set(profiles[0])
    state = read(root)
    state["active_provider"] = "openrouter"
    state["providers"][provider]["discovery"] = {"token_endpoint": "https://auth.x.ai/token"}
    root.write_text(json.dumps(state), encoding="utf-8")
    endpoint = install_endpoint(monkeypatch, provider)
    held = []
    endpoint.before_post = lambda _: held.append(getattr(A._auth_lock_holder_for(root), "depth", 0) > 0)
    resolve = A.resolve_codex_runtime_credentials if provider == "openai-codex" else A.resolve_xai_oauth_runtime_credentials
    result = resolve(force_refresh=True)
    assert held == [True]
    assert read(root)["active_provider"] == "openrouter"
    assert read(root)["providers"][provider]["tokens"]["access_token"] == result["api_key"]
    assert provider not in read(profiles[0])["providers"]


def test_classic_manual_alias_adopts_atomic_singleton_rotation(grant_stores, monkeypatch):
    # Existing supported-writer invariant: matching PREVIOUS tokens prove
    # lineage; merely sharing a file (or account identity) does not.
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    client = pool(root)
    tokens = {"access_token": "new-access", "refresh_token": "new-refresh"}
    save = A._save_codex_tokens if provider == "openai-codex" else A._save_xai_oauth_tokens
    save(tokens)
    endpoint = install_endpoint(monkeypatch, provider)
    result = client._refresh_entry(client.entries()[0], force=True)
    assert result is not None and result.refresh_token == "new-refresh"
    assert endpoint.calls == []


def test_same_store_independent_grant_is_not_singleton_alias(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, token="independent", singleton=False)
    state = read(root)
    state["providers"][provider] = {"tokens": {"access_token": "other-access", "refresh_token": "other-refresh"}}
    root.write_text(json.dumps(state), encoding="utf-8")
    client = pool(root)
    before = read(root)["providers"][provider]
    endpoint = install_endpoint(monkeypatch, provider)
    result = client._refresh_entry(client.entries()[0], force=True)
    assert result is not None
    assert endpoint.calls == ["independent"]
    assert read(root)["providers"][provider] == before


def test_root_and_profile_processes_share_refresh_lock(grant_stores, tmp_path):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    worker = tmp_path / "worker.py"
    worker.write_text("""
import json, os, sys, time
from pathlib import Path
from agent import credential_pool as CP
from hermes_cli import auth as A
assert Path(CP.__file__).resolve().is_relative_to(Path.cwd())
assert Path(A.__file__).resolve().is_relative_to(Path.cwd())
provider, root, active, barrier, name = sys.argv[1:]
root, active, barrier = Path(root), Path(active), Path(barrier)
A._auth_file_path = lambda: active
A._global_auth_file_path = CP._global_auth_file_path = lambda: None if active == root else root
CP._load_config_safe = lambda: {}
client = CP.load_pool(provider)
(barrier / name).touch()
deadline = time.monotonic() + 10
while not (barrier / 'go').exists():
    if time.monotonic() > deadline:
        raise TimeoutError('barrier')
    time.sleep(0.01)
def refresh(access, token):
    with (barrier / 'posts').open('a', encoding='utf-8') as log:
        log.write(token + '\\n')
    time.sleep(0.1)
    return {'access_token': 'access-next', 'refresh_token': 'next'}
setattr(A, 'refresh_codex_oauth_pure' if provider == 'openai-codex' else 'refresh_xai_oauth_pure', refresh)
result = client._refresh_entry(client.entries()[0], force=True)
assert result and result.refresh_token == 'next'
""", encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    processes = [subprocess.Popen([sys.executable, str(worker), provider, str(root), str(path), str(tmp_path), str(i)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 cwd=repo,
                                 env={**os.environ, "PYTHONPATH": str(repo), "HERMES_HOME": str(path.parent),
                                      "HOME": str(root.parent.parent)})
                 for i, path in enumerate((root, profiles[0]))]
    try:
        deadline = time.monotonic() + 10
        while not all((tmp_path / str(i)).exists() for i in range(2)):
            assert time.monotonic() < deadline, "workers did not load their stale rows"
            time.sleep(0.01)
        (tmp_path / "go").touch()
        for process in processes:
            stdout, stderr = process.communicate(timeout=15)
            assert process.returncode == 0, (stdout, stderr)
        assert (tmp_path / "posts").read_text(encoding="utf-8").splitlines() == ["original"]
        state = read(root)
        assert state["providers"][provider]["tokens"]["refresh_token"] == "next"
        assert state["credential_pool"][provider][0]["refresh_token"] == "next"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.communicate()


def test_new_login_stays_in_active_profile(grant_stores):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    before = read(root)
    active.set(profiles[0])
    save = A._save_codex_tokens if provider == "openai-codex" else A._save_xai_oauth_tokens
    save({"access_token": "independent-access", "refresh_token": "independent-refresh"})
    assert read(root) == before
    assert read(profiles[0])["providers"][provider]["tokens"]["refresh_token"] == "independent-refresh"


@pytest.mark.parametrize("shadow", [False, True])
def test_xai_non_pool_terminal_failure_clears_source(grant_stores, monkeypatch, shadow):
    provider, root, profiles, active, seed, pool = grant_stores
    if provider != "xai-oauth":
        pytest.skip("xAI runtime quarantine")
    seed(root, source="device_code" if shadow else "manual:device_code")
    state = read(root)
    if shadow:
        state["providers"][provider]["tokens"] = {}
    state["providers"][provider]["discovery"] = {"token_endpoint": "https://auth.x.ai/token"}
    root.write_text(json.dumps(state), encoding="utf-8")
    active.set(profiles[0])
    endpoint = install_endpoint(monkeypatch, provider)
    endpoint.consumed.add("original")
    with pytest.raises(A.AuthError):
        A.resolve_xai_oauth_runtime_credentials(force_refresh=True)
    assert not read(root)["providers"][provider]["tokens"].get("refresh_token")
    assert read(root)["providers"][provider]["last_auth_error"]
    row = read(root)["credential_pool"][provider][0]
    assert row["last_status"] == CP.STATUS_DEAD and row["last_status_at"]
    with pytest.raises(A.AuthError):
        A.resolve_xai_oauth_runtime_credentials(force_refresh=True)
    assert endpoint.calls == ["original"]
    assert provider not in read(profiles[0])["providers"]


def test_xai_non_pool_refresh_with_pool_only_root(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    if provider != "xai-oauth":
        pytest.skip("xAI runtime pool fallback")
    seed(root, singleton=False)
    active.set(profiles[0])
    endpoint = install_endpoint(monkeypatch, provider)
    monkeypatch.setattr(A, "_xai_oauth_discovery", lambda *args: {"token_endpoint": "https://auth.x.ai/token"})
    held = []
    endpoint.before_post = lambda _: held.append(getattr(A._auth_lock_holder_for(root), "depth", 0) > 0)
    result = A.resolve_xai_oauth_runtime_credentials(force_refresh=True)
    assert held == [True]
    assert read(root)["credential_pool"][provider][0]["access_token"] == result["api_key"]
    assert provider not in read(profiles[0])["providers"]


@pytest.mark.parametrize("root_home", [False, True])
def test_status_snapshot_preserves_rotated_and_removed_tokens(grant_stores, root_home):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=False)
    client = pool(root if root_home else profiles[0])
    state = read(root)
    row = state["credential_pool"][provider][0]
    row["refresh_token"] = "peer-refresh"
    row.pop("access_token")
    root.write_text(json.dumps(state), encoding="utf-8")
    client._mark_exhausted(client.entries()[0], 429)
    row = read(root)["credential_pool"][provider][0]
    assert row["refresh_token"] == "peer-refresh"
    assert not row.get("access_token")


@pytest.mark.parametrize("source", ["device_code", "manual:device_code"])
@pytest.mark.parametrize("shadow", [False, True])
def test_owned_row_never_seeds_or_refreshes_foreign_singleton(grant_stores, monkeypatch, source, shadow):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    seed(profiles[0], source=source, token="local", singleton=False)
    state = read(profiles[0])
    if shadow:
        state["providers"][provider] = {"tokens": {}, "last_auth_error": {"relogin_required": True}}
        profiles[0].write_text(json.dumps(state), encoding="utf-8")
    active.set(profiles[0])
    before = read(root)
    client = CP.load_pool(provider)
    local = next(entry for entry in client.entries() if entry.id == "local")
    assert local.refresh_token == "local"
    endpoint = install_endpoint(monkeypatch, provider)
    result = client._refresh_entry(local, force=True)
    assert result is not None and result.refresh_token == "local-next"
    assert endpoint.calls == ["local"]
    assert read(root) == before
    assert read(profiles[0])["credential_pool"][provider][0]["refresh_token"] == "local-next"


def test_profile_singleton_login_seeds_and_refreshes_its_own_row(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    profiles[0].write_text(json.dumps({"providers": {provider: {"tokens": {
        "access_token": "local-access", "refresh_token": "local-refresh"}}}}), encoding="utf-8")
    active.set(profiles[0])
    before = read(root)
    client = CP.load_pool(provider)
    assert len(client.entries()) == 1
    local = client.entries()[0]
    assert local.refresh_token == "local-refresh"
    assert read(profiles[0])["credential_pool"][provider][0]["id"] == local.id
    endpoint = install_endpoint(monkeypatch, provider)
    result = client._refresh_entry(local, force=True)
    assert result is not None and endpoint.calls == ["local-refresh"]
    assert read(root) == before


def test_borrowed_row_retains_source_after_another_process_adds_local_grant(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=False)
    active.set(profiles[0])
    client = CP.load_pool(provider)
    borrowed = client.entries()[0]
    seed(profiles[0], token="independent", singleton=False)
    local_before = read(profiles[0])
    endpoint = install_endpoint(monkeypatch, provider)
    result = client._refresh_entry(borrowed, force=True)
    assert result is not None and endpoint.calls == ["original"]
    assert read(root)["credential_pool"][provider][0]["refresh_token"] == "original-next"
    assert read(profiles[0]) == local_before


def test_removed_source_row_fails_before_consuming_refresh_token(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=False)
    active.set(profiles[0])
    client = CP.load_pool(provider)
    state = read(root)
    state["credential_pool"][provider] = []
    root.write_text(json.dumps(state), encoding="utf-8")
    endpoint = install_endpoint(monkeypatch, provider)
    assert client._refresh_entry(client.entries()[0], force=True) is None
    assert endpoint.calls == []
    assert client.entries()[0].last_status == CP.STATUS_EXHAUSTED


def test_borrowed_pool_add_switches_new_grant_to_active_store(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=False)
    active.set(profiles[0])
    client = CP.load_pool(provider)
    local = CP.PooledCredential(provider=provider, id="local", label="local", auth_type="oauth", priority=0,
                                source="manual:device_code", access_token="local-access", refresh_token="local-refresh")
    client.add_entry(local)
    before = read(root)
    endpoint = install_endpoint(monkeypatch, provider)
    result = client._refresh_entry(client.entries()[0], force=True)
    assert result is not None and endpoint.calls == ["local-refresh"]
    assert read(root) == before
    assert read(profiles[0])["credential_pool"][provider][0]["refresh_token"] == "local-refresh-next"


def test_borrowed_singleton_without_pool_rows_creates_source_row(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    state = read(root)
    state.pop("credential_pool")
    root.write_text(json.dumps(state), encoding="utf-8")
    active.set(profiles[0])
    client = CP.load_pool(provider)
    assert len(client.entries()) == 1
    endpoint = install_endpoint(monkeypatch, provider)
    result = client._refresh_entry(client.entries()[0], force=True)
    assert result is not None and endpoint.calls == ["original"]
    assert read(root)["credential_pool"][provider][0]["refresh_token"] == result.refresh_token
    assert not read(profiles[0]).get("credential_pool")


def test_device_row_without_singleton_adopts_peer_rotation(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(profiles[0], source="device_code", token="local", singleton=False)
    active.set(profiles[0])
    first, second = CP.load_pool(provider), CP.load_pool(provider)
    endpoint = install_endpoint(monkeypatch, provider)
    winner = first._refresh_entry(first.entries()[0], force=True)
    follower = second._refresh_entry(second.entries()[0], force=True)
    assert winner is not None and follower is not None
    assert follower.refresh_token == winner.refresh_token
    assert endpoint.calls == ["local"]


def test_singleton_reseed_reaches_its_existing_row_on_disk(grant_stores):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, source="device_code")
    active.set(root)
    state = read(root)
    state["providers"][provider]["tokens"] = {"access_token": "peer-access", "refresh_token": "peer-refresh"}
    root.write_text(json.dumps(state), encoding="utf-8")
    client = CP.load_pool(provider)
    assert client.entries()[0].refresh_token == "peer-refresh"
    assert read(root)["credential_pool"][provider][0]["refresh_token"] == "peer-refresh"


def test_loaded_root_pool_keeps_authority_when_active_context_changes(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=False)
    active.set(root)
    client = CP.load_pool(provider)
    active.set(profiles[0])
    endpoint = install_endpoint(monkeypatch, provider)
    result = client._refresh_entry(client.entries()[0], force=True)
    assert result is not None and endpoint.calls == ["original"]
    assert read(root)["credential_pool"][provider][0]["refresh_token"] == "original-next"
    assert not read(profiles[0]).get("credential_pool")


@pytest.mark.parametrize("rejected", [False, True])
def test_classic_three_references_share_one_rotation(grant_stores, monkeypatch, rejected):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, source="device_code")
    state = read(root)
    alias = dict(state["credential_pool"][provider][0], id="alias", source="manual:device_code", label="legacy alias", priority=1)
    state["credential_pool"][provider].append(alias)
    root.write_text(json.dumps(state), encoding="utf-8")
    active.set(root)
    client = CP.load_pool(provider)
    device = next(entry for entry in client.entries() if entry.id == "original")
    manual = next(entry for entry in client.entries() if entry.id == "alias")
    endpoint = install_endpoint(monkeypatch, provider)
    if rejected:
        def rotate_before_rejection(_):
            endpoint.before_post = None
            assert client._refresh_entry(device, force=True) is not None
        endpoint.before_post = rotate_before_rejection
    else:
        assert client._refresh_entry(device, force=True) is not None
    result = client._refresh_entry(manual, force=True)
    assert result is not None and result.refresh_token == "original-next"
    assert endpoint.calls == (["original", "original"] if rejected else ["original"])
    state = read(root)
    rows = {row["id"]: row for row in state["credential_pool"][provider]}
    assert rows["original"]["refresh_token"] == rows["alias"]["refresh_token"] == "original-next"
    assert rows["alias"]["label"] == "legacy alias" and rows["alias"]["priority"] == 1
    assert state["providers"][provider]["tokens"]["refresh_token"] == "original-next"


def test_borrower_does_not_recreate_suppressed_root_grant(grant_stores):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    state = read(root)
    state.pop("credential_pool")
    state["suppressed_sources"] = {provider: ["device_code"]}
    root.write_text(json.dumps(state), encoding="utf-8")
    before = read(root)
    active.set(profiles[0])
    assert CP.load_pool(provider).entries() == []
    assert read(root) == before


@pytest.mark.parametrize("counterpart", ["singleton", "sibling"])
@pytest.mark.parametrize("expired", [False, True])
def test_rejection_adopts_successor_with_shared_previous_access_token(grant_stores, monkeypatch, counterpart, expired):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=False)
    old_access = "access-original"
    if expired:
        import base64
        payload = base64.urlsafe_b64encode(json.dumps({"exp": 1}).encode()).decode().rstrip("=")
        old_access = f"header.{payload}.signature"
        state = read(root)
        state["credential_pool"][provider][0]["access_token"] = old_access
        root.write_text(json.dumps(state), encoding="utf-8")
    active.set(root)
    client = CP.load_pool(provider)
    endpoint = install_endpoint(monkeypatch, provider)
    def rotate_before_rejection(_):
        endpoint.before_post = None
        state = read(root)
        successor = {"access_token": old_access, "refresh_token": "peer-refresh"}
        if counterpart == "singleton":
            state["providers"][provider] = {"tokens": successor}
        else:
            state["credential_pool"][provider].append(dict(state["credential_pool"][provider][0], id="peer", **successor))
        root.write_text(json.dumps(state), encoding="utf-8")
        endpoint.consumed.add("original")
    endpoint.before_post = rotate_before_rejection
    result = client._refresh_entry(client.entries()[0], force=True)
    expected = "peer-refresh-next" if expired else "peer-refresh"
    assert result is not None and result.refresh_token == expected
    assert endpoint.calls == (["original", "peer-refresh"] if expired else ["original"])
    assert read(root)["credential_pool"][provider][0]["refresh_token"] == expected


@pytest.mark.parametrize("consumer", ["pool", "runtime"])
@pytest.mark.parametrize("root_home", [False, True])
def test_failed_rotation_commit_blocks_replay_in_fresh_process(grant_stores, monkeypatch, consumer, root_home):
    from collections import OrderedDict
    from agent import anthropic_credentials as AC

    monkeypatch.setattr(AC, "_SPENT_ROTATION_FINGERPRINTS", OrderedDict())
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, token="save-failure-grant")
    state = read(root)
    state["providers"][provider]["discovery"] = {"token_endpoint": "https://auth.x.ai/token"}
    root.write_text(json.dumps(state), encoding="utf-8")
    active.set(root if root_home else profiles[0])
    client = CP.load_pool(provider)
    client._current_id = client.entries()[0].id
    endpoint = install_endpoint(monkeypatch, provider)
    resolve = A.resolve_codex_runtime_credentials if provider == "openai-codex" else A.resolve_xai_oauth_runtime_credentials
    real_save = A._save_auth_store

    def broken_save(store, target_path=None):
        if target_path == root:
            raise OSError("simulated source write failure")
        return real_save(store, target_path=target_path)

    monkeypatch.setattr(A, "_save_auth_store", broken_save)
    monkeypatch.setattr(CP, "_save_auth_store", broken_save)
    if consumer == "pool":
        assert client._refresh_entry(client.entries()[0], force=True) is None
    else:
        with pytest.raises(A.AuthError) as failure:
            resolve(force_refresh=True)
        assert failure.value.code == "credential_persist_failed"
    assert client.current() is None
    assert client.peek() is None
    assert endpoint.calls == ["save-failure-grant"]
    with pytest.raises(A.AuthError) as failure:
        resolve(force_refresh=True)
    assert failure.value.code == "credential_persist_failed"
    assert endpoint.calls == ["save-failure-grant"]
    assert read(root)["credential_pool"][provider][0]["refresh_token"] == "save-failure-grant"

    # A separate interpreter has no in-memory fingerprint registry. Both
    # consumers must read the durable verdict at the shared authority.
    worker = r"""
import os, sys
from pathlib import Path
from agent import credential_pool as CP
from hermes_cli import auth as A
assert Path(CP.__file__).resolve().is_relative_to(Path.cwd())
assert Path(A.__file__).resolve().is_relative_to(Path.cwd())
provider, root, active = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
A._auth_file_path = lambda: active
A._global_auth_file_path = lambda: root
CP._global_auth_file_path = lambda: root
CP._load_config_safe = lambda: {}
def forbidden(*args, **kwargs):
    raise AssertionError("spent grant was posted again")
A.refresh_codex_oauth_pure = A.refresh_xai_oauth_pure = forbidden
client = CP.load_pool(provider)
assert client.peek() is None
assert client.select() is None
assert client._refresh_entry(client.entries()[0], force=True) is None
resolve = A.resolve_codex_runtime_credentials if provider == "openai-codex" else A.resolve_xai_oauth_runtime_credentials
try:
    resolve(force_refresh=True)
except A.AuthError as exc:
    assert exc.code == "credential_persist_failed"
else:
    raise AssertionError("spent grant was returned")
"""
    repo = Path(CP.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(repo), HERMES_HOME=str(profiles[1].parent), HOME=str(root.parent.parent))
    worker_active = root if root_home else profiles[1]
    env["HERMES_HOME"] = str(worker_active.parent)
    result = subprocess.run([sys.executable, "-c", worker, provider, str(root), str(worker_active)],
                            cwd=repo, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr

    if root_home and provider == "openai-codex":
        # A different grant may still serve the fallback; only spent material
        # is excluded, not the whole provider or store.
        monkeypatch.setattr(A, "_save_auth_store", real_save)
        state = read(root)
        state["credential_pool"][provider].append(dict(state["credential_pool"][provider][0],
            id="healthy", access_token="healthy-access", refresh_token="healthy-refresh"))
        root.write_text(json.dumps(state), encoding="utf-8")
        assert resolve(refresh_if_expiring=False)["api_key"] == "healthy-access"

    monkeypatch.setattr(A, "_save_auth_store", real_save)
    monkeypatch.setattr(CP, "_save_auth_store", real_save)
    save = A._save_codex_tokens if provider == "openai-codex" else A._save_xai_oauth_tokens
    save({"access_token": "reauth-access", "refresh_token": "reauth-refresh"}, source_path=root)
    selected = client.select()
    assert selected is not None and selected.access_token == "reauth-access"
    assert endpoint.calls == ["save-failure-grant"]


@pytest.mark.parametrize("failure", ["removed", "lock"])
def test_pending_refresh_failure_still_selects_healthy_sibling(grant_stores, monkeypatch, failure):
    import base64

    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=False)
    state = read(root)
    payload = base64.urlsafe_b64encode(json.dumps({"exp": 1}).encode()).decode().rstrip("=")
    state["credential_pool"][provider][0]["access_token"] = f"header.{payload}.signature"
    sibling = dict(state["credential_pool"][provider][0], id="healthy", access_token="healthy-access",
                   refresh_token="healthy-refresh", priority=1)
    state["credential_pool"][provider].append(sibling)
    root.write_text(json.dumps(state), encoding="utf-8")
    active.set(profiles[0])
    client = CP.load_pool(provider)
    endpoint = install_endpoint(monkeypatch, provider)
    if failure == "removed":
        state["credential_pool"][provider] = [sibling]
        root.write_text(json.dumps(state), encoding="utf-8")
    else:
        real_lock = CP._auth_store_lock
        def failed_lock(*args, **kwargs):
            if kwargs.get("target_path") == root:
                raise OSError("source lock unavailable")
            return real_lock(*args, **kwargs)
        monkeypatch.setattr(CP, "_auth_store_lock", failed_lock)
    assert client.select().id == "healthy"
    assert endpoint.calls == []


@pytest.mark.parametrize("successor", [False, True])
def test_rejection_save_failure_benches_in_memory(grant_stores, monkeypatch, successor):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    active.set(root)
    client = CP.load_pool(provider)
    entry = client.entries()[0]
    endpoint = install_endpoint(monkeypatch, provider)
    endpoint.consumed.add("original")
    if successor:
        def rotate_before_rejection(_):
            endpoint.before_post = None
            state = read(root)
            state["providers"][provider]["tokens"]["refresh_token"] = "peer-refresh"
            root.write_text(json.dumps(state), encoding="utf-8")
        endpoint.before_post = rotate_before_rejection
    def broken_save(*args, **kwargs):
        raise OSError("source save unavailable")
    monkeypatch.setattr(CP, "_save_auth_store", broken_save)
    assert client._refresh_entry(entry, force=True) is None
    assert client.entries()[0].last_status in {CP.STATUS_DEAD, CP.STATUS_EXHAUSTED}
    assert endpoint.calls == ["original"]


@pytest.mark.parametrize("source", ["device_code", "manual:device_code"])
def test_terminal_grant_is_not_reported_or_returned_by_runtime(grant_stores, monkeypatch, source):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, source=source)
    active.set(root)
    client = CP.load_pool(provider)
    endpoint = install_endpoint(monkeypatch, provider)
    endpoint.consumed.add("original")
    assert client._refresh_entry(client.entries()[0], force=True) is None
    assert client.select() is None
    assert A.get_auth_status(provider)["logged_in"] is False
    resolve = A.resolve_codex_runtime_credentials if provider == "openai-codex" else A.resolve_xai_oauth_runtime_credentials
    with pytest.raises(A.AuthError):
        resolve(refresh_if_expiring=False)
    assert endpoint.calls == ["original"]


@pytest.mark.parametrize("existing", ["empty", "api_key", "manual"])
def test_borrower_projects_live_singleton_beside_other_rows(grant_stores, existing):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    state = read(root)
    rows = []
    if existing != "empty":
        rows = [dict(state["credential_pool"][provider][0], id="independent", source="manual:device_code",
                     access_token="independent-access", refresh_token="independent-refresh")]
        if existing == "api_key":
            rows[0].update(auth_type="api_key", source="manual:api_key", access_token="independent-key", refresh_token="")
    state["credential_pool"][provider] = rows
    root.write_text(json.dumps(state), encoding="utf-8")
    active.set(profiles[0])
    client = CP.load_pool(provider)
    assert any(entry.source == "device_code" and entry.refresh_token == "original" for entry in client.entries())
    persisted = read(root)["credential_pool"][provider]
    assert any(row["source"] == "device_code" and row["refresh_token"] == "original" for row in persisted)
    if rows:
        assert next(row for row in persisted if row["id"] == "independent") == rows[0]
    assert not read(profiles[0]).get("credential_pool")


def test_xai_runtime_rotation_preserves_unrelated_device_grant(grant_stores, monkeypatch):
    provider, root, profiles, active, seed, pool = grant_stores
    if provider != "xai-oauth":
        pytest.skip("xAI saver sibling fan-out")
    seed(root, source="device_code")
    state = read(root)
    state["providers"][provider] = {"tokens": {}, "last_auth_error": "old incident"}
    sibling = dict(state["credential_pool"][provider][0], id="independent", access_token="independent-access",
                   refresh_token="independent-refresh", priority=1, label="other account", last_status="exhausted",
                   last_status_at=123, last_error_reason="quota", last_error_reset_at=time.time() + 3600)
    state["credential_pool"][provider].append(sibling)
    root.write_text(json.dumps(state), encoding="utf-8")
    active.set(root)
    endpoint = install_endpoint(monkeypatch, provider)
    result = A.resolve_xai_oauth_runtime_credentials(force_refresh=True)
    assert result["api_key"] == "access-original-next"
    rows = read(root)["credential_pool"][provider]
    assert rows[0]["refresh_token"] == "original-next"
    assert rows[1] == sibling
    assert endpoint.calls == ["original"]


def test_forced_runtime_refresh_adopts_fresh_peer_without_post(grant_stores, monkeypatch):
    from hermes_cli import auth_xai

    provider, root, profiles, active, seed, pool = grant_stores
    seed(root)
    active.set(profiles[0])
    endpoint = install_endpoint(monkeypatch, provider)
    owner, name = (A, "_read_codex_tokens") if provider == "openai-codex" else (auth_xai, "_read_xai_oauth_tokens")
    real_read = getattr(owner, name)
    reads = 0
    def read_then_peer_rotates(**kwargs):
        nonlocal reads
        data = real_read(**kwargs)
        reads += 1
        if reads == 1:
            with A._auth_store_lock(target_path=root):
                store = A._load_auth_store(root)
                store["providers"][provider]["tokens"] = {
                    "access_token": "peer-access", "refresh_token": "peer-refresh"}
                A._save_auth_store(store, target_path=root)
        return data
    monkeypatch.setattr(owner, name, read_then_peer_rotates)
    resolve = A.resolve_codex_runtime_credentials if provider == "openai-codex" else A.resolve_xai_oauth_runtime_credentials
    assert resolve(force_refresh=True)["api_key"] == "peer-access"
    assert reads >= 2 and endpoint.calls == []


@pytest.mark.parametrize("projection", [False, True])
def test_borrowed_load_save_failure_keeps_usable_pool(grant_stores, monkeypatch, caplog, projection):
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, source="device_code")
    state = read(root)
    if projection:
        state.pop("credential_pool")
    else:
        state["providers"][provider]["tokens"] = {"access_token": "peer-access", "refresh_token": "peer-refresh"}
    root.write_text(json.dumps(state), encoding="utf-8")
    before = read(root)
    profile_before = read(profiles[0])
    active.set(profiles[0])
    real_save = A._save_auth_store
    def broken_root_save(store, target_path=None):
        if target_path == root:
            raise OSError("root store is read-only")
        return real_save(store, target_path=target_path)
    monkeypatch.setattr(A, "_save_auth_store", broken_root_save)
    monkeypatch.setattr(CP, "_save_auth_store", broken_root_save)
    with caplog.at_level("WARNING", logger=CP.__name__):
        client = CP.load_pool(provider)
    assert client.peek().access_token == ("access-original" if projection else "peer-access")
    assert caplog.records and any(record.levelname == "WARNING" for record in caplog.records)
    assert read(root) == before and read(profiles[0]) == profile_before


def test_in_lock_sync_rejects_consumed_uncommitted_disk_pair(grant_stores, monkeypatch):
    from collections import OrderedDict
    from agent import anthropic_credentials as AC

    monkeypatch.setattr(AC, "_SPENT_ROTATION_FINGERPRINTS", OrderedDict())
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=False)
    active.set(profiles[0])
    client = CP.load_pool(provider)
    entry = client.entries()[0]
    with A._auth_store_lock(target_path=root):
        state = read(root)
        state["credential_pool"][provider][0].update(access_token="spent-peer-access", refresh_token="spent-peer-refresh")
        root.write_text(json.dumps(state), encoding="utf-8")
        AC.mark_rotation_consumed_uncommitted("spent-peer-access", "spent-peer-refresh", source_path=root)
    endpoint = install_endpoint(monkeypatch, provider)
    assert client._refresh_entry(entry, force=True) is None
    assert endpoint.calls == []
    assert client.entries()[0].last_error_reason == "credential_persist_failed"


def test_peek_revives_persist_failed_manual_row_after_peer_reauth(grant_stores, monkeypatch):
    from collections import OrderedDict
    from agent import anthropic_credentials as AC

    monkeypatch.setattr(AC, "_SPENT_ROTATION_FINGERPRINTS", OrderedDict())
    provider, root, profiles, active, seed, pool = grant_stores
    seed(root, singleton=False)
    state = read(root)
    state["credential_pool"][provider][0].update(last_status=CP.STATUS_DEAD, last_status_at=time.time(),
                                              last_error_reason="credential_persist_failed")
    root.write_text(json.dumps(state), encoding="utf-8")
    with A._auth_store_lock(target_path=root):
        AC.mark_rotation_consumed_uncommitted("access-original", "original", source_path=root)
    active.set(profiles[0])
    client = CP.load_pool(provider)
    assert client.peek() is None
    with A._auth_store_lock(target_path=root):
        state["credential_pool"][provider][0].update(access_token="reauth-access", refresh_token="reauth-refresh",
                                                  last_status=None, last_error_reason=None)
        root.write_text(json.dumps(state), encoding="utf-8")
    assert client.peek().access_token == "reauth-access"
