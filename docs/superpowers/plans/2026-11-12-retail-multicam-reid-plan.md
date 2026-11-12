# Retail Multi-Camera Re-ID Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mini retail video-analytics pipeline (detection → per-camera tracking → cross-camera Re-ID → unique-visitor counting/dwell-time → FastAPI metrics) validated on public multi-camera footage and a short self-recorded clip, with a GPU optimization benchmark, in one week.

**Architecture:** One Python process runs an async worker per camera (VideoSource → YOLOv8 detection → ByteTrack tracking → periodic Re-ID embedding). A central IdentityResolver matches embeddings across cameras by cosine similarity within a temporal window, assigning global visitor IDs. Everything persists to SQLite; FastAPI exposes read endpoints over it.

**Tech Stack:** Python, `uv`, `ultralytics` (YOLOv8 + ByteTrack/BoT-SORT), `torchreid`/OSNet (fallback: ResNet50 pooling), `pydantic` v2, `sqlite3`, `FastAPI`, `httpx` (functional tests), `pytest` + `pytest.mark.parametrize`, TensorRT/`onnxruntime-gpu`, Docker.

**Spec:** [docs/superpowers/specs/2026-11-12-retail-multicam-reid-design.md](../specs/2026-11-12-retail-multicam-reid-design.md)

## Global Constraints

- Hardware: RTX 3050, 6GB VRAM — use YOLOv8 n/s only, never l/x.
- Stretch pose-estimation step runs as a separate pass, never concurrent with the main pipeline, to avoid VRAM contention.
- TensorRT export is attempted first; if it doesn't build cleanly in ~1 hour, fall back to ONNX Runtime with CUDA Execution Provider — both are acceptable optimization backends to benchmark.
- Re-ID backbone: `torchreid`/OSNet preferred; if it doesn't install cleanly, fall back to `fast-reid`, then to ResNet50 (ImageNet-pretrained) global-average-pool embeddings as last resort — documented as a conscious degradation, not hidden.
- Validation data: EPFL CVLAB terrace/passageway sequences (primary), WILDTRACK via Kaggle CC0 mirror (optional extension). **DukeMTMC is forbidden** — retracted for privacy violations.
- Persistence is SQLite only — no Postgres/Redis/Kafka.
- Single process, async task per camera — no distributed messaging layer.
- Windows host — no DeepStream.
- Type hints on every public function; `logging`, never `print()`; specific exceptions, no bare `except Exception`; `pathlib` for all paths.

## Review Focus

- A false-positive detection (reflection, poster, mannequin) must not spawn a persistent global identity that survives more than one frame — test in Task 5 (IdentityResolver) with a single-sighting embedding never repeated.
- A person who leaves and re-enters the *same* camera's frame within the temporal window must resolve back to their existing global_id, not get treated only as a cross-camera case — test in Task 5.
- Two different people with visually similar clothing (near-threshold cosine similarity) must not silently merge into one global_id without it showing up in evaluation — test in Task 5 (similarity just above/below threshold) and Task 9 (evaluation reports false-merge rate).
- A person reappearing *after* the temporal window has closed must get a new global_id, not match a stale gallery entry — test in Task 5.
- Running two camera workers plus periodic Re-ID embedding on a 6GB card must not crash on OOM; the pipeline must degrade (skip an embedding cycle) rather than throw unhandled — test in Task 8 (pipeline) with a forced low-VRAM simulation path.

---

## File Structure

```
src/retail_cv/
  config.py                 # pydantic Settings (thresholds, paths, model names)
  video_source.py           # Frame, VideoSource (file or camera index)
  detection.py               # Detection, PersonDetector
  tracking.py                 # TrackedBox, CameraTracker (ByteTrack/BoT-SORT)
  reid/embedder.py             # ReIDEmbedder (OSNet, fallback chain)
  identity_resolver.py          # IdentityMatch, IdentityResolver  <- Juan Diego escribe la lógica
  persistence/store.py           # Store (SQLite)
  datasets/epfl_terrace.py        # download_and_prepare(), ground truth parsing
  pipeline.py                      # RetailPipeline orchestrator
  evaluation/validate_reid.py       # EvalReport, evaluate()
  api/app.py                         # FastAPI app + endpoints
  optimize/benchmark.py               # BenchmarkResult, benchmark_backend()
  interaction/pose_heuristic.py        # ProductInteractionDetector <- Juan Diego escribe la lógica (stretch)
tests/
  conftest.py
  test_video_source.py
  test_detection.py
  test_tracking.py
  test_reid_embedder.py
  test_identity_resolver.py
  test_store.py
  test_pipeline.py
  test_evaluation.py
  test_api.py
  test_pose_heuristic.py            # stretch
docker/Dockerfile
docker/docker-compose.yml
data/samples/                        # tiny committed fixtures (few-second clips/frames)
data/raw/                            # downloaded datasets, gitignored
```

---

### Task 1: Project scaffolding + VideoSource

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/retail_cv/__init__.py`, `src/retail_cv/config.py`, `src/retail_cv/video_source.py`
- Test: `tests/conftest.py`, `tests/test_video_source.py`

**Interfaces:**
- Produces: `retail_cv.video_source.Frame` (pydantic model: `image: np.ndarray`, `frame_index: int`, `timestamp_s: float`, `camera_id: str`; `model_config = ConfigDict(arbitrary_types_allowed=True)`); `retail_cv.video_source.VideoSource(source: str | int, camera_id: str, fps_override: float | None = None)` with method `frames(self) -> Iterator[Frame]`.

- [ ] **Step 1: Init project**

Run: `uv init --name retail-cv` then `uv add ultralytics opencv-python pydantic fastapi httpx sqlalchemy --dev pytest ruff` (adjust exact deps as Tasks 4/11 require; `torchreid`/`onnxruntime-gpu`/`tensorrt` get added when their tasks need them, to isolate install failures to the task they belong to).

- [ ] **Step 2: Add a 3-second sample clip fixture**

Place a short (~3s, single person walking) test clip at `data/samples/sample_single_person.mp4` (record it yourself with the webcam, or trim a few seconds from the EPFL terrace footage once Task 7 downloads it — a placeholder of any short mp4 with motion is enough to unblock this task).

- [ ] **Step 3: Write the failing test**

```python
def test_video_source_yields_frames_with_camera_id(sample_clip_path):
    source = VideoSource(str(sample_clip_path), camera_id="zone_a")
    frames = list(source.frames())
    assert len(frames) > 0
    assert all(f.camera_id == "zone_a" for f in frames)
    assert frames[0].frame_index == 0
    assert frames[-1].frame_index == len(frames) - 1
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_video_source.py -v`
Expected: FAIL with `ImportError` or `NameError`.

- [ ] **Step 5: Implement `VideoSource`**

Wrap `cv2.VideoCapture(source)` (accepts both a file path string and an int camera index, same call). `frames()` reads until `cap.read()` returns `False`, yielding `Frame` with an incrementing `frame_index` and `timestamp_s = frame_index / fps` (use `cap.get(cv2.CAP_PROP_FPS)`, or `fps_override` if given — needed for the EPFL sequences, whose FPS metadata may be missing or wrong).

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_video_source.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml .gitignore src/retail_cv tests/conftest.py tests/test_video_source.py data/samples
git commit -m "feat: project scaffolding and VideoSource abstraction"
```

---

### Task 2: PersonDetector

**Files:**
- Create: `src/retail_cv/detection.py`
- Test: `tests/test_detection.py`

**Interfaces:**
- Consumes: `Frame.image` from Task 1 (an `np.ndarray`, BGR as OpenCV returns it).
- Produces: `Detection` (pydantic: `bbox_xyxy: tuple[float, float, float, float]`, `confidence: float`, `class_id: int`); `PersonDetector(model_path: str = "yolov8n.pt", confidence_threshold: float = 0.4)` with `detect(self, image: np.ndarray) -> list[Detection]`.

- [ ] **Step 1: Add a fixture frame with a known person**

Use a single frame extracted from `data/samples/sample_single_person.mp4` (via `cv2.VideoCapture` + `.read()` in `conftest.py`, fixture name `frame_with_person`).

- [ ] **Step 2: Write the failing test**

```python
def test_detect_finds_person_above_threshold(frame_with_person):
    detector = PersonDetector(confidence_threshold=0.4)
    detections = detector.detect(frame_with_person)
    assert len(detections) >= 1
    assert all(d.class_id == 0 for d in detections)  # COCO person class
    assert all(d.confidence >= 0.4 for d in detections)

def test_detect_returns_empty_on_blank_frame():
    detector = PersonDetector()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detector.detect(blank) == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_detection.py -v`
Expected: FAIL (`ImportError`/`NameError`).

- [ ] **Step 4: Implement `PersonDetector`**

`YOLO(model_path)` from `ultralytics`, filtered to `classes=[0]` (person) at inference time, mapped to `Detection` objects.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_detection.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/retail_cv/detection.py tests/test_detection.py
git commit -m "feat: YOLOv8 person detector wrapper"
```

---

### Task 3: CameraTracker (ByteTrack)

**Files:**
- Create: `src/retail_cv/tracking.py`
- Test: `tests/test_tracking.py`

**Interfaces:**
- Consumes: `np.ndarray` frames (same as `PersonDetector.detect`).
- Produces: `TrackedBox` (pydantic: `track_id: int`, `bbox_xyxy: tuple[float, float, float, float]`, `confidence: float`); `CameraTracker(model_path: str = "yolov8n.pt", tracker_cfg: Literal["bytetrack.yaml", "botsort.yaml"] = "bytetrack.yaml")` with `update(self, image: np.ndarray) -> list[TrackedBox]` (stateful — call once per frame in order).

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.parametrize("tracker_cfg", ["bytetrack.yaml", "botsort.yaml"])
def test_track_id_stable_across_consecutive_frames(sample_clip_frames, tracker_cfg):
    tracker = CameraTracker(tracker_cfg=tracker_cfg)
    all_ids = [tb.track_id for frame in sample_clip_frames[:10] for tb in tracker.update(frame.image)]
    assert len(set(all_ids)) <= 2  # one person in the clip; allow one ID switch at most
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tracking.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `CameraTracker`**

Use `ultralytics` `model.track(frame, persist=True, tracker=tracker_cfg, classes=[0])` under the hood; map `result.boxes` (with `.id`) to `TrackedBox`, skipping boxes where `.id is None` (untracked detections).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tracking.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retail_cv/tracking.py tests/test_tracking.py
git commit -m "feat: per-camera ByteTrack/BoT-SORT tracker wrapper"
```

---

### Task 4: ReIDEmbedder

**Files:**
- Create: `src/retail_cv/reid/__init__.py`, `src/retail_cv/reid/embedder.py`
- Test: `tests/test_reid_embedder.py`

**Interfaces:**
- Consumes: a cropped person image (`np.ndarray`, from `TrackedBox.bbox_xyxy` applied to a frame — cropping itself happens in the pipeline, Task 8, not here).
- Produces: `ReIDEmbedder(backend: Literal["osnet", "fastreid", "resnet50_fallback"] = "osnet")` with `embed(self, crop: np.ndarray) -> np.ndarray` returning an L2-normalized vector of `self.embedding_dim` floats; `embedding_dim: int` attribute set per backend.

- [ ] **Step 1: Try installing `torchreid`**

Run: `uv add torchreid`. If it fails to install or import cleanly against the project's torch/CUDA versions (per Global Constraints fallback chain), try `uv add fast-reid`; if that also fails, skip both — `resnet50_fallback` only needs `torchvision`, already a transitive dep of `ultralytics`.

- [ ] **Step 2: Write the failing tests**

```python
def test_embed_returns_l2_normalized_vector_of_expected_dim(person_crop):
    embedder = ReIDEmbedder()
    vec = embedder.embed(person_crop)
    assert vec.shape == (embedder.embedding_dim,)
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-4

def test_embed_same_crop_twice_is_near_identical(person_crop):
    embedder = ReIDEmbedder()
    v1, v2 = embedder.embed(person_crop), embedder.embed(person_crop)
    assert np.dot(v1, v2) > 0.99

def test_embed_different_people_is_less_similar(person_crop, other_person_crop):
    embedder = ReIDEmbedder()
    v1 = embedder.embed(person_crop)
    v2 = embedder.embed(other_person_crop)
    assert np.dot(v1, v2) < 0.95  # loose bound: different crops shouldn't look identical
```

`other_person_crop` fixture: a crop of a visibly different person (different clothing colors at minimum) — grab one from the EPFL terrace sample once Task 7 lands, or a stock photo crop checked into `data/samples/`.

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_reid_embedder.py -v`
Expected: FAIL.

- [ ] **Step 4: Implement `ReIDEmbedder`**

Load the chosen backend's pretrained weights (OSNet via `torchreid.models.build_model(name="osnet_x0_25", pretrained=True)`, or the fallback chain); resize crop to the backend's expected input size; run inference; L2-normalize the output feature vector.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_reid_embedder.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/retail_cv/reid tests/test_reid_embedder.py pyproject.toml uv.lock
git commit -m "feat: Re-ID embedder with fallback chain"
```

---

### Task 5: IdentityResolver — tú escribes la lógica

**Files:**
- Create: `src/retail_cv/identity_resolver.py`
- Test: `tests/test_identity_resolver.py`

**Interfaces:**
- Consumes: `embedding: np.ndarray` (L2-normalized, `ReIDEmbedder.embedding_dim` from Task 4).
- Produces: `IdentityMatch` (dataclass: `global_id: str`, `is_new: bool`); `IdentityResolver(similarity_threshold: float = 0.6, temporal_window_s: float = 300.0)` with `resolve(self, camera_id: str, local_track_id: int, embedding: np.ndarray, timestamp_s: float) -> IdentityMatch`.

Esta es la pieza central del proyecto — la que responde "¿cómo evito contar dos veces al mismo cliente en 20 cámaras?" en la entrevista. El esqueleto y los tests van abajo; el cuerpo de `resolve()` lo escribes tú.

- [ ] **Step 1: Write the failing tests (exact behavior contract)**

```python
def test_first_sighting_creates_new_global_id():
    resolver = IdentityResolver()
    match = resolver.resolve("cam_a", local_track_id=1, embedding=EMB_PERSON_1, timestamp_s=0.0)
    assert match.is_new is True

def test_same_embedding_within_window_resolves_to_same_global_id():
    resolver = IdentityResolver(temporal_window_s=300.0)
    first = resolver.resolve("cam_a", 1, EMB_PERSON_1, timestamp_s=0.0)
    second = resolver.resolve("cam_b", 7, EMB_PERSON_1, timestamp_s=120.0)  # different camera/track, same person
    assert second.global_id == first.global_id
    assert second.is_new is False

def test_different_embedding_above_threshold_distance_creates_new_global_id():
    resolver = IdentityResolver(similarity_threshold=0.6)
    first = resolver.resolve("cam_a", 1, EMB_PERSON_1, timestamp_s=0.0)
    second = resolver.resolve("cam_a", 2, EMB_PERSON_2, timestamp_s=5.0)  # cosine sim < 0.6, fixture-defined
    assert second.global_id != first.global_id

def test_same_embedding_after_window_closes_creates_new_global_id():
    resolver = IdentityResolver(temporal_window_s=60.0)
    first = resolver.resolve("cam_a", 1, EMB_PERSON_1, timestamp_s=0.0)
    second = resolver.resolve("cam_a", 1, EMB_PERSON_1, timestamp_s=61.0)
    assert second.global_id != first.global_id  # Review Focus: stale gallery entry must not match

def test_person_reentering_same_camera_within_window_keeps_global_id():
    resolver = IdentityResolver(temporal_window_s=300.0)
    first = resolver.resolve("cam_a", 1, EMB_PERSON_1, timestamp_s=0.0)
    # local_track_id changes (tracker lost and re-acquired them) but it's the same camera, same person
    second = resolver.resolve("cam_a", 9, EMB_PERSON_1, timestamp_s=30.0)
    assert second.global_id == first.global_id  # Review Focus: same-camera re-entry

def test_near_threshold_similar_clothing_is_distinguishable_with_fixture_embeddings():
    # EMB_PERSON_1_SIMILAR_CLOTHES is fixture-crafted to sit just below the default threshold
    resolver = IdentityResolver(similarity_threshold=0.6)
    first = resolver.resolve("cam_a", 1, EMB_PERSON_1, timestamp_s=0.0)
    second = resolver.resolve("cam_b", 2, EMB_PERSON_1_SIMILAR_CLOTHES, timestamp_s=10.0)
    assert second.global_id != first.global_id  # Review Focus: near-threshold false-merge guard
```

`EMB_PERSON_1`, `EMB_PERSON_2`, `EMB_PERSON_1_SIMILAR_CLOTHES`: fixed `np.ndarray` constants in `conftest.py`, hand-crafted (e.g. random unit vectors with a controlled cosine distance between them) so the tests don't depend on the real embedder — this test suite is about the resolver's matching logic, not about Task 4's model quality.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_identity_resolver.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `IdentityResolver.resolve()`**

Diseña tú la galería interna (qué guardas por `global_id`: ¿el último embedding? ¿un promedio? ¿los últimos N?) y la regla de matching (cosine similarity contra qué candidatos, cómo rompes empates si dos global_ids superan el umbral). El contrato que no puedes romper: los 6 tests de arriba definen el comportamiento exacto esperado — si tu diseño interno los pasa, es válido.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_identity_resolver.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retail_cv/identity_resolver.py tests/test_identity_resolver.py
git commit -m "feat: cross-camera identity resolver"
```

---

### Task 6: Store (SQLite persistence)

**Files:**
- Create: `src/retail_cv/persistence/__init__.py`, `src/retail_cv/persistence/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: `IdentityMatch.global_id` (Task 5), `camera_id`, `timestamp_s`, `bbox_xyxy` from the pipeline.
- Produces: `Store(db_path: str = ":memory:")` with `record_sighting(self, global_id: str, camera_id: str, timestamp_s: float, bbox_xyxy: tuple[float, float, float, float]) -> None`, `unique_visitor_count(self, since_s: float | None = None) -> int`, `dwell_time_s(self, global_id: str, camera_id: str) -> float`, `visitor_path(self, global_id: str) -> list[dict]` (each dict: `camera_id`, `first_seen_s`, `last_seen_s`).

- [ ] **Step 1: Write the failing tests**

```python
def test_unique_visitor_count_deduplicates_by_global_id():
    store = Store()
    store.record_sighting("g1", "cam_a", 0.0, (0, 0, 10, 10))
    store.record_sighting("g1", "cam_b", 5.0, (0, 0, 10, 10))
    store.record_sighting("g2", "cam_a", 1.0, (0, 0, 10, 10))
    assert store.unique_visitor_count() == 2

def test_dwell_time_is_span_between_first_and_last_sighting_in_zone():
    store = Store()
    store.record_sighting("g1", "cam_a", 0.0, (0, 0, 10, 10))
    store.record_sighting("g1", "cam_a", 12.5, (0, 0, 10, 10))
    assert store.dwell_time_s("g1", "cam_a") == pytest.approx(12.5)

def test_visitor_path_lists_zones_in_order_of_first_sighting():
    store = Store()
    store.record_sighting("g1", "cam_b", 5.0, (0, 0, 10, 10))
    store.record_sighting("g1", "cam_a", 0.0, (0, 0, 10, 10))
    path = store.visitor_path("g1")
    assert [p["camera_id"] for p in path] == ["cam_a", "cam_b"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `Store`**

Single table `sightings(global_id TEXT, camera_id TEXT, timestamp_s REAL, bbox_xyxy TEXT)` via stdlib `sqlite3`; aggregate with `GROUP BY`/`MIN`/`MAX` for the three read methods.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retail_cv/persistence tests/test_store.py
git commit -m "feat: SQLite persistence for sightings and dwell time"
```

---

### Task 7: EPFL terrace dataset ingestion

**Files:**
- Create: `src/retail_cv/datasets/__init__.py`, `src/retail_cv/datasets/epfl_terrace.py`
- Test: `tests/test_epfl_terrace.py`

**Interfaces:**
- Produces: `download_and_prepare(dest_dir: Path, sequence: Literal["terrace1", "passageway1"] = "terrace1") -> list[Path]` (per-camera video/frame-sequence paths, usable directly as `VideoSource` sources from Task 1); `load_ground_truth(dest_dir: Path, sequence: str) -> list[GroundTruthSighting]` where `GroundTruthSighting` (pydantic): `person_id: str`, `camera_id: str`, `frame_index: int`, `bbox_xyxy: tuple[float, float, float, float]`.

- [ ] **Step 1: Manually verify the download**

From `cvlab.epfl.ch/data/pom`, download the `terrace1` sequence (4 cameras, ~5000 frames, research-use license) into `data/raw/epfl_terrace1/`. Inspect the actual file/annotation format on disk before writing the parser — the plan can't pin the exact format sight-unseen; adapt `GroundTruthSighting` parsing to whatever ships (frame-indexed bounding boxes per camera with person IDs is the expected shape per the dataset paper).

- [ ] **Step 2: Write the failing test**

```python
def test_prepare_returns_one_path_per_camera(tmp_path):
    paths = download_and_prepare(tmp_path, sequence="terrace1")
    assert len(paths) == 4
    assert all(p.exists() for p in paths)

def test_ground_truth_has_consistent_person_ids_across_cameras(tmp_path):
    download_and_prepare(tmp_path, sequence="terrace1")
    gt = load_ground_truth(tmp_path, sequence="terrace1")
    person_ids = {g.person_id for g in gt}
    cameras_per_person = {pid: {g.camera_id for g in gt if g.person_id == pid} for pid in person_ids}
    assert any(len(cams) > 1 for cams in cameras_per_person.values())  # at least one person seen on 2+ cameras
```

Mark both `@pytest.mark.slow` (network download) and skip in default CI-style runs; run explicitly during this task.

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_epfl_terrace.py -v -m slow`
Expected: FAIL.

- [ ] **Step 4: Implement `download_and_prepare` and `load_ground_truth`**

Download once, cache under `dest_dir`; convert whatever raw format ships (frames or video) into one playable file per camera; parse the provided calibration/ground-truth annotations into `GroundTruthSighting` rows.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_epfl_terrace.py -v -m slow`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/retail_cv/datasets tests/test_epfl_terrace.py
git commit -m "feat: EPFL terrace dataset ingestion and ground truth parsing"
```

(`data/raw/` stays gitignored — only the ingestion code is committed, not the dataset itself.)

---

### Task 8: RetailPipeline orchestrator

**Files:**
- Create: `src/retail_cv/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `VideoSource` (Task 1), `PersonDetector` (Task 2), `CameraTracker` factory (Task 3, one instance per camera — trackers are stateful), `ReIDEmbedder` (Task 4), `IdentityResolver` (Task 5), `Store` (Task 6).
- Produces: `RetailPipeline(sources: list[VideoSource], detector: PersonDetector, tracker_factory: Callable[[], CameraTracker], embedder: ReIDEmbedder, resolver: IdentityResolver, store: Store, embed_every_n_frames: int = 10, min_track_age_frames: int = 5)` with `async def run(self, max_frames_per_camera: int | None = None) -> None`. `min_track_age_frames` exists specifically so a transient single-frame false-positive detection (reflection, poster) never reaches the resolver/store as a counted visitor — a track must be seen for at least that many consecutive frames before its first embedding/sighting is recorded.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.asyncio
async def test_pipeline_populates_store_with_at_least_one_visitor(two_camera_sample_sources, detector, embedder, store):
    pipeline = RetailPipeline(two_camera_sample_sources, detector, CameraTracker, embedder, IdentityResolver(), store)
    await pipeline.run(max_frames_per_camera=30)
    assert store.unique_visitor_count() >= 1

@pytest.mark.asyncio
async def test_pipeline_ignores_transient_track_shorter_than_min_age(two_camera_sample_sources, detector, embedder, store, monkeypatch):
    # fake tracker that yields track_id=1 on frame 0 only, then nothing — simulates a one-frame false positive
    class OneFrameTracker:
        def __init__(self): self._done = False
        def update(self, image):
            if self._done: return []
            self._done = True
            return [TrackedBox(track_id=1, bbox_xyxy=(0, 0, 10, 10), confidence=0.9)]
    pipeline = RetailPipeline(two_camera_sample_sources, detector, OneFrameTracker, embedder, IdentityResolver(), store, min_track_age_frames=5)
    await pipeline.run(max_frames_per_camera=10)
    assert store.unique_visitor_count() == 0

@pytest.mark.asyncio
async def test_pipeline_skips_embedding_cycle_on_simulated_oom(two_camera_sample_sources, detector, store, monkeypatch):
    embedder = ReIDEmbedder()
    def raise_oom(*_args, **_kwargs):
        raise torch.cuda.OutOfMemoryError("simulated")
    monkeypatch.setattr(embedder, "embed", raise_oom)
    pipeline = RetailPipeline(two_camera_sample_sources, detector, CameraTracker, embedder, IdentityResolver(), store)
    await pipeline.run(max_frames_per_camera=10)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `RetailPipeline.run()`**

One `asyncio.create_task` per `VideoSource`; each task owns its own `CameraTracker` instance (from `tracker_factory`) and a `dict[int, int]` of `track_id -> consecutive_frame_count`. Only once a `track_id` has been seen for `min_track_age_frames` consecutive frames does it start calling `embedder.embed()` (every `embed_every_n_frames` frames from that point), feeding the result into `resolver.resolve()`, and writing to `store.record_sighting()`. Wrap the `embedder.embed()` call in a `try/except torch.cuda.OutOfMemoryError` that logs and skips that cycle (Review Focus item on graceful VRAM degradation) instead of propagating.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retail_cv/pipeline.py tests/test_pipeline.py
git commit -m "feat: async multi-camera pipeline orchestrator"
```

---

### Task 9: Evaluation against ground truth

**Files:**
- Create: `src/retail_cv/evaluation/__init__.py`, `src/retail_cv/evaluation/validate_reid.py`
- Test: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `Store.visitor_path`/raw sightings (Task 6), `GroundTruthSighting` list (Task 7).
- Produces: `EvalReport` (pydantic: `precision: float`, `recall: float`, `id_switches: int`, `false_merge_count: int`); `evaluate(predicted_sightings: list[tuple[str, str, int]], ground_truth: list[GroundTruthSighting]) -> EvalReport` where predicted tuples are `(global_id, camera_id, frame_index)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_perfect_match_gives_precision_recall_1():
    gt = [GroundTruthSighting(person_id="p1", camera_id="cam_a", frame_index=0, bbox_xyxy=(0,0,1,1))]
    predicted = [("g1", "cam_a", 0)]
    report = evaluate(predicted, gt)
    assert report.precision == 1.0
    assert report.recall == 1.0

def test_two_predicted_ids_for_one_true_person_counts_as_id_switch():
    gt = [
        GroundTruthSighting(person_id="p1", camera_id="cam_a", frame_index=0, bbox_xyxy=(0,0,1,1)),
        GroundTruthSighting(person_id="p1", camera_id="cam_b", frame_index=1, bbox_xyxy=(0,0,1,1)),
    ]
    predicted = [("g1", "cam_a", 0), ("g2", "cam_b", 1)]  # resolver failed to merge them
    report = evaluate(predicted, gt)
    assert report.id_switches >= 1

def test_one_predicted_id_for_two_true_people_counts_as_false_merge():
    gt = [
        GroundTruthSighting(person_id="p1", camera_id="cam_a", frame_index=0, bbox_xyxy=(0,0,1,1)),
        GroundTruthSighting(person_id="p2", camera_id="cam_b", frame_index=1, bbox_xyxy=(0,0,1,1)),
    ]
    predicted = [("g1", "cam_a", 0), ("g1", "cam_b", 1)]  # resolver wrongly merged them
    report = evaluate(predicted, gt)
    assert report.false_merge_count >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `evaluate()`**

Match `(camera_id, frame_index)` keys between predicted and ground truth to build a predicted-id → true-id contingency table; `id_switches` = true IDs mapped to more than one predicted ID, `false_merge_count` = predicted IDs mapped to more than one true ID; `precision`/`recall` from the majority-mapping per predicted ID (standard cluster-purity style metric — document the exact formula used in a docstring, since it's a simplified stand-in for IDF1/HOTA).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_evaluation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retail_cv/evaluation tests/test_evaluation.py
git commit -m "feat: Re-ID evaluation against ground truth"
```

---

### Task 10: FastAPI backend

**Files:**
- Create: `src/retail_cv/api/__init__.py`, `src/retail_cv/api/app.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `Store` (Task 6) as a FastAPI dependency.
- Produces: `app: FastAPI` with `GET /visitors/unique?since_s=<float|null>` → `{"count": int}`; `GET /visitors/{global_id}` → `{"global_id": str, "path": list[dict], "dwell_time_s": dict[str, float]}` (404 if unknown); `GET /health` → `{"status": "ok"}`.

- [ ] **Step 1: Write the failing functional tests**

```python
@pytest.mark.asyncio
async def test_unique_visitors_endpoint_returns_count(app_with_seeded_store):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_with_seeded_store), base_url="http://test") as client:
        resp = await client.get("/visitors/unique")
        assert resp.status_code == 200
        assert resp.json() == {"count": 2}

@pytest.mark.asyncio
async def test_visitor_detail_404_for_unknown_id(app_with_seeded_store):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app_with_seeded_store), base_url="http://test") as client:
        resp = await client.get("/visitors/does-not-exist")
        assert resp.status_code == 404
```

`app_with_seeded_store` fixture: builds the FastAPI app with a `Store` pre-populated via `record_sighting` calls (reuses Task 6's `Store` directly, no mocking — functional test against the real endpoint per project convention).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_api.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the endpoints**

Standard FastAPI route handlers calling into `Store`; `HTTPException(404)` when `visitor_path` returns empty.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retail_cv/api tests/test_api.py
git commit -m "feat: FastAPI metrics endpoints"
```

---

### Task 11: Docker packaging

**Files:**
- Create: `docker/Dockerfile`, `docker/docker-compose.yml`

**Interfaces:**
- Consumes: the full `src/retail_cv` package (all prior tasks) and `pyproject.toml`.
- Produces: a container exposing the FastAPI app (Task 10) on port 8000, with GPU passthrough configured (`--gpus all` / `deploy.resources.reservations.devices` in compose) for the detector/tracker/embedder to use CUDA.

- [ ] **Step 1: Write the Dockerfile**

Base on an official `nvidia/cuda` runtime image matching the project's installed CUDA version (check with `nvidia-smi` on the dev machine first); `uv sync --frozen`; `CMD ["uv", "run", "uvicorn", "retail_cv.api.app:app", "--host", "0.0.0.0", "--port", "8000"]`.

- [ ] **Step 2: Write `docker-compose.yml`**

One service, GPU reservation, volume-mount `data/` for the SQLite file and sample clips so the container doesn't need its own copy baked in.

- [ ] **Step 3: Verify it builds and runs**

Run: `docker compose -f docker/docker-compose.yml build && docker compose -f docker/docker-compose.yml up -d`
Expected: container starts; `curl http://localhost:8000/health` returns `{"status": "ok"}`.

- [ ] **Step 4: Commit**

```bash
git add docker/
git commit -m "feat: Docker packaging with GPU passthrough"
```

---

### Task 12: GPU optimization benchmark

**Files:**
- Create: `src/retail_cv/optimize/__init__.py`, `src/retail_cv/optimize/export.py`, `src/retail_cv/optimize/benchmark.py`
- Test: `tests/test_benchmark.py` (smoke test only — this task's real output is a report, not pass/fail assertions)

**Interfaces:**
- Produces: `export_model(model_path: str, backend: Literal["tensorrt", "onnxruntime"]) -> Path` (exported artifact path; raises `RuntimeError` with a clear message if TensorRT export fails, per the Global Constraints fallback); `BenchmarkResult` (pydantic: `backend: str`, `num_streams: int`, `avg_fps: float`, `peak_vram_mb: float`); `benchmark_backend(model_path_or_engine: str, backend: Literal["pytorch", "tensorrt", "onnxruntime"], num_streams: int, num_frames: int = 200) -> BenchmarkResult`.

- [ ] **Step 1: Implement `export_model()`**

Wrap `YOLO(model_path).export(format="engine")`; catch the exception, log it, and retry with `format="onnx"` (for use with `onnxruntime-gpu`) per the Global Constraints fallback. Return the resulting artifact `Path` either way, and record which backend actually succeeded (an attribute or log line — Task 14's writeup needs to know which path was used).

- [ ] **Step 2: Run `export_model("yolov8n.pt", backend="tensorrt")` manually**

Time-box to ~1 hour. Confirm which backend actually produced the artifact (TensorRT engine or ONNX fallback) before moving on — this determines which `backend` values Step 6 benchmarks.

- [ ] **Step 3: Write the smoke test**

```python
def test_benchmark_result_has_positive_fps(sample_clip_frames):
    result = benchmark_backend("yolov8n.pt", backend="pytorch", num_streams=1, num_frames=20)
    assert result.avg_fps > 0
    assert result.peak_vram_mb > 0
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: FAIL.

- [ ] **Step 5: Implement `benchmark_backend`**

Run inference in a loop over `num_frames` (looping the sample clip if shorter), `num_streams` concurrent `asyncio`/thread workers for the dual-stream case; measure wall time for `avg_fps`; `torch.cuda.max_memory_allocated()` for `peak_vram_mb`.

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_benchmark.py -v`
Expected: PASS

- [ ] **Step 7: Produce the actual comparison numbers**

Run `benchmark_backend` for `(pytorch, 1)`, `(pytorch, 2)`, `(tensorrt_or_onnxruntime, 1)`, `(tensorrt_or_onnxruntime, 2)`; save results to `docs/writeup/benchmark_results.json` — this table is what goes in the final README.

- [ ] **Step 8: Commit**

```bash
git add src/retail_cv/optimize tests/test_benchmark.py docs/writeup/benchmark_results.json
git commit -m "feat: GPU optimization benchmark (PyTorch vs TensorRT/ONNX Runtime)"
```

---

### Task 13 (stretch): Product interaction heuristic — tú escribes la lógica

Solo si los Tasks 1-12 están listos con margen de días.

**Files:**
- Create: `src/retail_cv/interaction/__init__.py`, `src/retail_cv/interaction/pose_heuristic.py`
- Test: `tests/test_pose_heuristic.py`

**Interfaces:**
- Consumes: pose keypoints per frame (from `ultralytics` YOLOv8-pose output — wrist keypoints specifically) and a `global_id` from the pipeline.
- Produces: `InteractionEvent` (pydantic: `global_id: str`, `zone_name: str`, `event_type: Literal["pick_up"]`, `frame_index: int`); `ProductInteractionDetector(zones: dict[str, tuple[float,float,float,float]], hold_frames: int = 15)` with `update(self, global_id: str, wrist_xy: tuple[float, float], frame_index: int) -> InteractionEvent | None`.

- [ ] **Step 1: Write the failing tests (behavior contract)**

```python
def test_wrist_inside_zone_for_hold_frames_emits_event():
    detector = ProductInteractionDetector(zones={"shelf": (0, 0, 10, 10)}, hold_frames=3)
    events = [detector.update("g1", (5, 5), frame_index=i) for i in range(5)]
    assert any(e is not None for e in events)

def test_wrist_passing_through_zone_briefly_emits_nothing():
    detector = ProductInteractionDetector(zones={"shelf": (0, 0, 10, 10)}, hold_frames=10)
    events = [detector.update("g1", (5, 5), frame_index=i) for i in range(3)]  # fewer frames than hold_frames
    assert all(e is None for e in events)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_pose_heuristic.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement `ProductInteractionDetector.update()`**

Tú decides cómo trackear el conteo de frames consecutivos por `(global_id, zone_name)` y cuándo exactamente disparar el evento (¿solo al cruzar el umbral `hold_frames`, o cada frame mientras se mantiene? — los 2 tests de arriba no distinguen eso, decídelo tú y documenta por qué).

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pose_heuristic.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/retail_cv/interaction tests/test_pose_heuristic.py
git commit -m "feat: product pick-up heuristic (stretch goal)"
```

---

### Task 14: Write-up and demo

**Files:**
- Create: `README.md`, `docs/writeup/interview_answers.md`

- [ ] **Step 1: Record the live demo**

Short clip (webcam + ELP) through the actual pipeline (Task 8), showing the unique-visitor count and dwell time changing as someone moves between the two cameras.

- [ ] **Step 2: Write `README.md`**

Architecture diagram (reuse the one from the spec), setup instructions (`uv sync`, how to run the pipeline, how to run the API, how to run the benchmark), and the benchmark table from Task 12.

- [ ] **Step 3: Write `docs/writeup/interview_answers.md`**

Answer, with concrete numbers from this project, the 4 filter questions from the job post: unique-visitor counting across cameras, pick-up detection without confusing a passerby, maintaining identity across camera gaps, and processing cost at scale (extrapolate from the Task 12 benchmark: FPS per stream × GPU → estimated GPUs needed for 20 cameras/location × 300 locations, stated as an order-of-magnitude estimate, not a false-precision number).

- [ ] **Step 4: Commit**

```bash
git add README.md docs/writeup
git commit -m "docs: README, architecture writeup, and interview answers"
```
