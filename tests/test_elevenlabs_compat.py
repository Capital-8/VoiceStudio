"""ElevenLabs-compatible API under /elevenlabs (api.routers.elevenlabs_compat).

Pinned: output formats (incl. resampled PCM, μ-law and A-law), voice lookup,
the ElevenLabs error shape, voice/model lists, speech-to-text words, and the
gapless streaming resampler.
"""
from __future__ import annotations

import importlib
import os
import uuid

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

os.environ.setdefault("OMNIVOICE_MODEL", "test")
os.environ.setdefault("OMNIVOICE_DISABLE_FILE_LOG", "1")

SECONDS = 0.5


def _el():
    return importlib.import_module("api.routers.elevenlabs_compat")


@pytest.fixture()
def engine(monkeypatch):
    tts = importlib.import_module("services.tts_backend")

    class _Engine(tts.TTSBackend):
        id = "elevenlabs-compat-engine"
        display_name = "ElevenLabs compat engine (test)"
        gpu_compat = ("cpu",)
        calls: list = []

        @property
        def sample_rate(self) -> int:
            return 24000

        @property
        def supported_languages(self) -> list[str]:
            return ["multi"]

        @classmethod
        def is_available(cls):
            return True, "ready"

        def generate(self, text, **kw):
            type(self).calls.append(kw)
            t = torch.arange(int(24000 * SECONDS)) / 24000
            return (0.3 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0)

    _Engine.calls = []
    inst = _Engine()
    monkeypatch.setitem(tts._REGISTRY, _Engine.id, _Engine)
    monkeypatch.setattr(tts, "get_active_tts_backend", lambda: inst)
    monkeypatch.setattr(tts, "get_engine_instance_for", lambda _id: inst)
    from services import engine_routing

    async def _profile(*_a, **_k):
        return {"routing_status": "native", "routing_reason": ""}

    monkeypatch.setattr(engine_routing, "runtime_compute_profile_async", _profile)
    return _Engine


@pytest.fixture()
def client():
    from main import app

    return TestClient(app, client=("127.0.0.1", 50000))


@pytest.fixture()
def profile():
    from core.db import db_conn, init_db

    init_db()
    pid = str(uuid.uuid4())
    name = f"el-{uuid.uuid4().hex[:8]}"
    with db_conn() as conn:
        conn.execute(
            "INSERT INTO voice_profiles (id, name, kind, created_at, ref_text, ref_audio_path, language) "
            "VALUES (?, ?, 'clone', 0, '', ?, 'en')",
            (pid, name, f"{pid}.wav"),
        )
    yield pid, name
    with db_conn() as conn:
        conn.execute("DELETE FROM voice_profiles WHERE id=?", (pid,))


def _tts(client, voice="default", fmt=None, stream=False, **body):
    path = f"/elevenlabs/v1/text-to-speech/{voice}" + ("/stream" if stream else "")
    params = {"output_format": fmt} if fmt else None
    return client.post(path, params=params, json={"text": "Hello there.", **body})


@pytest.mark.parametrize("fmt,rate,width", [
    ("pcm_24000", 24000, 2), ("pcm_16000", 16000, 2), ("pcm_44100", 44100, 2),
    ("ulaw_8000", 8000, 1), ("alaw_8000", 8000, 1),
])
@pytest.mark.parametrize("stream", [False, True])
def test_raw_formats_have_the_requested_rate(client, engine, fmt, rate, width, stream):
    res = _tts(client, fmt=fmt, stream=stream, model_id="eleven_multilingual_v2")
    assert res.status_code == 200, res.text
    samples = len(res.content) // width
    assert abs(samples - rate * SECONDS) <= 2
    assert res.headers["content-type"].startswith(_el()._MEDIA[fmt.split("_")[0]])


def test_wav_and_profile_voice(client, engine, profile):
    pid, _ = profile
    res = _tts(client, voice=pid, fmt="wav_44100", language_code="en",
               voice_settings={"stability": 0.4, "similarity_boost": 0.8, "speed": 1.1}, seed=7)
    assert res.status_code == 200, res.text
    assert res.content[:4] == b"RIFF"
    kw = engine.calls[-1]
    assert kw["ref_audio"].endswith(f"{pid}.wav")
    assert kw["speed"] == pytest.approx(1.1)
    assert kw["seed"] == 7
    assert kw["language"] == "en"


def test_unknown_elevenlabs_voice_id_is_a_404(client, engine):
    res = _tts(client, voice="21m00Tcm4TlvDq8ikWAM", fmt="pcm_24000")
    assert res.status_code == 404
    detail = res.json()["detail"]
    assert detail["status"] == "voice_not_found"
    assert "/elevenlabs/v1/voices" in detail["message"]
    assert engine.calls == []


def test_bad_output_format_is_rejected_before_generation(client, engine):
    res = _tts(client, fmt="flac_48000")
    assert res.status_code == 422
    assert res.json()["detail"]["status"] == "invalid_output_format"
    assert engine.calls == []


def test_voice_and_model_lists(client, engine, profile):
    pid, name = profile
    voices = client.get("/elevenlabs/v1/voices").json()["voices"]
    entry = next(v for v in voices if v["voice_id"] == pid)
    assert entry["name"] == name and entry["category"] == "cloned"
    assert voices[0]["voice_id"] == "default"

    assert client.get(f"/elevenlabs/v1/voices/{pid}").json()["name"] == name
    missing = client.get("/elevenlabs/v1/voices/nope")
    assert missing.status_code == 404 and missing.json()["detail"]["status"] == "voice_not_found"

    page = client.get("/elevenlabs/v2/voices", params={"search": name, "page_size": 1}).json()
    assert [v["voice_id"] for v in page["voices"]] == [pid]
    assert page["has_more"] is False and page["total_count"] == 1

    assert client.get("/elevenlabs/v1/voices/settings/default").json()["similarity_boost"] == 0.75
    ids = [m["model_id"] for m in client.get("/elevenlabs/v1/models").json()]
    assert "eleven_multilingual_v2" in ids


def test_speech_to_text_returns_elevenlabs_words(client, monkeypatch):
    oc = importlib.import_module("api.routers.openai_compat")
    seen = {}

    async def fake(**kw):
        seen.update(kw)
        return {"text": "Hi there", "language": "en",
                "words": [{"word": "Hi", "start": 0.0, "end": 0.3},
                          {"word": "there", "start": 0.4, "end": 0.8}]}

    monkeypatch.setattr(oc, "_transcribe_request", fake)
    res = client.post("/elevenlabs/v1/speech-to-text",
                      data={"model_id": "scribe_v1", "language_code": "en"},
                      files={"file": ("a.wav", b"RIFF", "audio/wav")})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["text"] == "Hi there" and body["language_code"] == "en"
    assert [w["type"] for w in body["words"]] == ["word", "spacing", "word"]
    assert body["words"][1] == {"text": " ", "start": 0.3, "end": 0.4, "type": "spacing", "logprob": 0.0}
    assert seen["response_format"] == "verbose_json" and seen["language"] == "en"



def test_speech_to_text_labels_speakers(client, monkeypatch):
    oc = importlib.import_module("api.routers.openai_compat")
    td = importlib.import_module("services.transcript_diarization")
    seen = {}

    async def fake(**kw):
        seen["upload"] = kw["file"].file.read()
        return {"text": "Hi there friend", "language": "en",
                "words": [{"word": "Hi", "start": 0.0, "end": 0.3},
                          {"word": "there", "start": 0.4, "end": 0.8},
                          {"word": "friend", "start": 1.0, "end": 1.4}]}

    def fake_diarize(path, spans, num_speakers):
        seen["path"] = path
        seen["kept"] = open(path, "rb").read()
        seen["spans"], seen["n"] = spans, num_speakers
        return ["SPEAKER_07", "SPEAKER_07", "SPEAKER_02"], None

    monkeypatch.setattr(oc, "_transcribe_request", fake)
    monkeypatch.setattr(td, "diarize_segments", fake_diarize)
    res = client.post("/elevenlabs/v1/speech-to-text", data={"diarize": "true", "num_speakers": "2"},
                      files={"file": ("a.wav", b"RIFF-audio", "audio/wav")})
    assert res.status_code == 200, res.text
    words = res.json()["words"]
    assert [w.get("speaker_id") for w in words if w["type"] == "word"] == ["speaker_0", "speaker_0", "speaker_1"]
    assert seen["upload"] == seen["kept"] == b"RIFF-audio"
    assert seen["n"] == 2 and len(seen["spans"]) == 3
    assert not os.path.exists(seen["path"])  # the kept copy is removed

    monkeypatch.setattr(td, "diarize_segments", lambda *a: ([None] * 3, "LOAD_FAILED"))
    res = client.post("/elevenlabs/v1/speech-to-text", data={"diarize": "true"},
                      files={"file": ("a.wav", b"RIFF-audio", "audio/wav")})
    assert res.status_code == 200
    assert res.headers["x-omnivoice-diarization-error"] == "LOAD_FAILED"
    assert all("speaker_id" not in w for w in res.json()["words"])


def test_audio_isolation_returns_the_vocal_stem(client, monkeypatch):
    import soundfile as sf

    vs = importlib.import_module("services.video_voice_swap")
    seen = {}

    async def fake_separate(track, workdir):
        seen["source"] = open(track, "rb").read()
        seen["track"] = track
        vocals = os.path.join(workdir, "vocals.wav")
        sf.write(vocals, np.zeros((44100, 2), dtype=np.float32), 44100)
        return vocals, vocals

    monkeypatch.setattr(vs, "separation_available", lambda: True)
    monkeypatch.setattr(vs, "separate_vocals", fake_separate)
    res = client.post("/elevenlabs/v1/audio-isolation", params={"output_format": "pcm_16000"},
                      files={"audio": ("song.mp3", b"song-bytes", "audio/mpeg")})
    assert res.status_code == 200, res.text
    assert len(res.content) == 16000 * 2
    assert seen["source"] == b"song-bytes" and not os.path.exists(seen["track"])

    monkeypatch.setattr(vs, "separation_available", lambda: False)
    res = client.post("/elevenlabs/v1/audio-isolation", params={"output_format": "pcm_16000"},
                      files={"audio": ("song.mp3", b"x", "audio/mpeg")})
    assert res.status_code == 409
    assert res.json()["detail"]["status"] == "audio_isolation_unavailable"


@pytest.mark.parametrize("rate", [8000, 16000, 22050, 44100, 48000])
def test_stream_resampler_matches_whole_signal(rate):
    import torchaudio.functional as AF

    rng = np.random.default_rng(rate)
    pcm = (rng.standard_normal(24000) * 6000).clip(-32768, 32767).astype("<i2")
    whole = AF.resample(torch.from_numpy(pcm.astype(np.float32) / 32768.0), 24000, rate).numpy()
    rs = _el().StreamResampler(rate)
    raw = pcm.tobytes()
    parts, i = [], 0
    for size in (1, 333, 4097, 2, 9000, 77):  # odd sizes split int16 samples
        parts.append(rs.feed(raw[i:i + size]))
        i += size
    parts.append(rs.feed(raw[i:]))
    parts.append(rs.flush())
    out = np.concatenate(parts).astype(np.float32) / 32768.0
    assert len(out) == len(whole)
    assert np.max(np.abs(out - whole)) < 2 / 32768.0


def test_alaw_matches_reference_codec():
    audioop = pytest.importorskip("audioop")
    pcm = np.arange(-32768, 32768, dtype=np.int16)
    assert _el().lin2alaw(pcm) == audioop.lin2alaw(pcm.astype("<i2").tobytes(), 2)


@pytest.fixture()
def saved_take():
    """A 1 s 24 kHz take in OUTPUTS_DIR, as the app's handlers leave one."""
    import soundfile as sf
    from core.config import OUTPUTS_DIR

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    name = f"el-take-{uuid.uuid4().hex[:8]}.wav"
    path = os.path.join(OUTPUTS_DIR, name)
    sf.write(path, np.zeros(24000, dtype=np.float32), 24000)
    yield {"audio_url": f"/audio/{name}", "id": "x"}
    os.unlink(path)


def test_speech_to_speech_keeps_delivery_then_falls_back(client, monkeypatch, profile, saved_take):
    from fastapi import HTTPException

    vc = importlib.import_module("api.routers.voice_convert")
    calls = []

    async def fake(*, audio, profile_id, match_duration, language, keep_delivery):
        calls.append((profile_id, keep_delivery, await audio.read()))
        if keep_delivery:
            raise HTTPException(409, {"error": "voice_conversion_engine_missing", "message": "none"})
        return saved_take

    monkeypatch.setattr(vc, "convert_speech", fake)
    pid, name = profile
    res = client.post(f"/elevenlabs/v1/speech-to-speech/{name}",
                      params={"output_format": "pcm_16000"},
                      files={"audio": ("in.wav", b"source-bytes", "audio/wav")})
    assert res.status_code == 200, res.text
    assert len(res.content) == 16000 * 2
    assert res.headers["x-omnivoice-fallback"] == "respoken"
    # The profile is resolved by name; the upload is re-read for the fallback.
    assert calls == [(pid, True, b"source-bytes"), (pid, False, b"source-bytes")]

    missing = client.post("/elevenlabs/v1/speech-to-speech/default",
                          files={"audio": ("in.wav", b"x", "audio/wav")})
    assert missing.status_code == 404
    assert missing.json()["detail"]["status"] == "voice_not_found"


def test_sound_generation_returns_the_saved_effect(client, monkeypatch, saved_take):
    from fastapi import HTTPException

    sfx = importlib.import_module("api.routers.sfx")
    seen = []

    async def fake(req):
        seen.append(req)
        if req.prompt == "missing":
            raise HTTPException(409, {"code": "sfx_not_installed", "message": "Install it."})
        return saved_take

    monkeypatch.setattr(sfx, "sfx_generate", fake)
    monkeypatch.setattr(sfx, "_limits", lambda: {"default_seconds": 10.0})
    res = client.post("/elevenlabs/v1/sound-generation", params={"output_format": "wav_44100"},
                      json={"text": "door creak", "prompt_influence": 0.3})
    assert res.status_code == 200, res.text
    assert res.content[:4] == b"RIFF"
    assert seen[-1].prompt == "door creak" and seen[-1].duration == 10.0

    err = client.post("/elevenlabs/v1/sound-generation", params={"output_format": "pcm_24000"},
                      json={"text": "missing", "duration_seconds": 2})
    assert err.status_code == 409
    detail = err.json()["detail"]
    assert detail["status"] == "sfx_not_installed" and detail["message"] == "Install it."


def test_text_to_dialogue_joins_each_voice_with_a_pause(client, engine, profile):
    pid, _ = profile
    res = client.post("/elevenlabs/v1/text-to-dialogue", params={"output_format": "pcm_24000"},
                      json={"inputs": [{"text": "Hi.", "voice_id": "default"},
                                       {"text": "Hello!", "voice_id": pid}],
                            "model_id": "eleven_v3", "seed": 3})
    assert res.status_code == 200, res.text
    gap = int(24000 * _el()._TURN_GAP_S)
    assert len(res.content) // 2 == 2 * int(24000 * SECONDS) + gap
    first, second = engine.calls[-2:]
    assert "ref_audio" not in first and second["ref_audio"].endswith(f"{pid}.wav")
    assert first["seed"] == second["seed"] == 3

    streamed = client.post("/elevenlabs/v1/text-to-dialogue/stream", params={"output_format": "pcm_24000"},
                           json={"inputs": [{"text": "Hi.", "voice_id": "default"}]})
    assert streamed.status_code == 200 and len(streamed.content) // 2 == int(24000 * SECONDS)


def test_text_to_dialogue_checks_every_voice_first(client, engine):
    res = client.post("/elevenlabs/v1/text-to-dialogue", params={"output_format": "pcm_24000"},
                      json={"inputs": [{"text": "Hi.", "voice_id": "default"},
                                       {"text": "Yo.", "voice_id": "21m00Tcm4TlvDq8ikWAM"}]})
    assert res.status_code == 404
    assert res.json()["detail"]["status"] == "voice_not_found"
    assert engine.calls == []


def test_with_timestamps_uses_recognized_words_when_available(client, engine, monkeypatch):
    import base64

    oc = importlib.import_module("api.routers.openai_compat")
    asr = importlib.import_module("services.asr_backend")
    monkeypatch.setattr(asr, "asr_model_missing_error", lambda: None)

    async def fake(**kw):
        assert kw["file"].file.read(4) == b"RIFF"
        return {"text": "Hello there.", "words": [{"word": "Hello", "start": 0.05, "end": 0.2},
                                                  {"word": "there", "start": 0.25, "end": 0.45}]}

    monkeypatch.setattr(oc, "_transcribe_request", fake)
    res = client.post("/elevenlabs/v1/text-to-speech/default/with-timestamps",
                      params={"output_format": "pcm_16000"}, json={"text": "Hello there."})
    assert res.status_code == 200, res.text
    assert res.headers["x-omnivoice-alignment"] == "recognized"
    body = res.json()
    assert len(base64.b64decode(body["audio_base64"])) == 16000 * 2 * SECONDS
    alignment = body["alignment"]
    assert "".join(alignment["characters"]) == "Hello there."
    assert alignment["character_start_times_seconds"][6] == pytest.approx(0.25)


def test_with_timestamps_estimates_without_a_recognizer(client, engine, monkeypatch):
    asr = importlib.import_module("services.asr_backend")
    monkeypatch.setattr(asr, "asr_model_missing_error", lambda: {"error": "asr_model_missing"})
    res = client.post("/elevenlabs/v1/text-to-speech/default/stream/with-timestamps",
                      params={"output_format": "pcm_24000"}, json={"text": "Hi you"})
    assert res.status_code == 200, res.text
    assert res.headers["x-omnivoice-alignment"] == "estimated"
    import json as _json

    line = _json.loads(res.content.decode().strip())
    ends = line["alignment"]["character_end_times_seconds"]
    assert len(ends) == 6 and ends[-1] == pytest.approx(SECONDS, abs=0.01)


def test_add_voice_creates_a_clone_profile(client, monkeypatch):
    pr = importlib.import_module("api.routers.profiles")
    seen = {}

    async def fake_create(**kw):
        seen.update(kw)
        seen["audio"] = await kw["ref_audio"].read()
        return {"id": "new-profile"}

    monkeypatch.setattr(pr, "create_profile", fake_create)
    res = client.post("/elevenlabs/v1/voices/add",
                      data={"name": "Narrator", "labels": '{"language": "en"}'},
                      files=[("files", ("a.wav", b"first", "audio/wav")),
                             ("files", ("b.wav", b"second", "audio/wav"))])
    assert res.status_code == 200, res.text
    assert res.json() == {"voice_id": "new-profile", "requires_verification": False}
    assert seen["name"] == "Narrator" and seen["kind"] == "clone"
    assert seen["language"] == "en" and seen["audio"] == b"first"

    bad = client.post("/elevenlabs/v1/voices/add", data={"name": "X", "labels": "[1]"},
                      files=[("files", ("a.wav", b"x", "audio/wav"))])
    assert bad.status_code == 422 and bad.json()["detail"]["status"] == "invalid_labels"


def test_delete_voice_removes_the_profile(client, monkeypatch, profile):
    pr = importlib.import_module("api.routers.profiles")
    deleted = []
    monkeypatch.setattr(pr, "delete_profile", lambda pid: deleted.append(pid) or {"deleted": pid})
    pid, name = profile
    res = client.delete(f"/elevenlabs/v1/voices/{name}")
    assert res.status_code == 200, res.text
    assert deleted == [pid]
    missing = client.delete("/elevenlabs/v1/voices/default")
    assert missing.status_code == 404


def test_deleting_a_voice_needs_administrator_access():
    from api.routers.elevenlabs_compat import router

    route = next(r for r in router.routes if r.path.endswith("/v1/voices/{voice_id}") and "DELETE" in r.methods)
    assert any(getattr(d.dependency, "__name__", "") == "require_admin" for d in route.dependencies)


def test_dubbing_runs_on_the_batch_queue(client, monkeypatch, tmp_path):
    batch = importlib.import_module("api.routers.batch")
    seen = {}

    async def fake_enqueue(**kw):
        seen.update(kw)
        seen["video"] = await kw["video"].read()
        return {"job_id": "dub123", "status": "queued", "queue_position": 1}

    monkeypatch.setattr(batch, "enqueue_batch_job", fake_enqueue)
    res = client.post("/elevenlabs/v1/dubbing", data={"target_lang": "es", "drop_background_audio": "true"},
                      files={"file": ("clip.mp4", b"video", "video/mp4")})
    assert res.status_code == 200, res.text
    assert res.json()["dubbing_id"] == "dub123"
    assert seen["langs"] == "es" and seen["preserve_bg"] is False and seen["video"] == b"video"

    no_file = client.post("/elevenlabs/v1/dubbing", data={"target_lang": "es", "source_url": "https://x"})
    assert no_file.status_code == 422 and no_file.json()["detail"]["status"] == "file_required"

    output = tmp_path / "out.mp4"
    output.write_bytes(b"dubbed-video")
    job = {"id": "dub123", "status": "running", "filename": "clip.mp4", "langs": ["es"],
           "outputs": {"es": str(output)}, "error": None}
    monkeypatch.setitem(batch._jobs, "dub123", job)
    status = client.get("/elevenlabs/v1/dubbing/dub123").json()
    assert status == {"dubbing_id": "dub123", "name": "clip.mp4", "status": "dubbing",
                      "target_languages": ["es"], "error": None}
    early = client.get("/elevenlabs/v1/dubbing/dub123/audio/es")
    assert early.status_code == 425 and early.json()["detail"]["status"] == "dubbing_not_ready"

    job["status"] = "done"
    assert client.get("/elevenlabs/v1/dubbing/dub123").json()["status"] == "dubbed"
    done = client.get("/elevenlabs/v1/dubbing/dub123/audio/es")
    assert done.status_code == 200 and done.content == b"dubbed-video"

    job.update(status="failed", error="Translation failed")
    assert client.get("/elevenlabs/v1/dubbing/dub123").json()["error"] == "Translation failed"
    assert client.get("/elevenlabs/v1/dubbing/nope").status_code == 404
