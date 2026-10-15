"""Exercise search navigation across Streamlit reruns with a small real FAISS index."""
from pathlib import Path
from types import SimpleNamespace

import faiss
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.common import config
from src.search import encode
from src.search import backends as search_backends
from src.search.search import Index


def test_search_similar_history_and_image_tab(tmp_path, monkeypatch):
    # Synthetic images keep the test independent of BDD100K and model weights.
    meta_dir = tmp_path / "meta"
    index_root = tmp_path / "indexes"
    meta_dir.mkdir()
    index_root.mkdir()
    for name in ["siglip384__whole", "siglip384__t3x2o20"]:
        folder = index_root / name
        folder.mkdir()
        (folder / "index.faiss").touch()
    ids = [f"test_{i:02d}" for i in range(20)]
    images = []
    for i, iid in enumerate(ids):
        name = iid + ".png"
        Image.new("RGB", (128, 72), (100 + i, 50, 20)).save(tmp_path / name)
        images.append({"image_id": iid, "image_path": name, "weather": "rainy" if i % 2 else "clear",
                       "scene": "city street", "timeofday": "night"})
    pd.DataFrame(images).to_parquet(meta_dir / "images.parquet", index=False)
    unc_dir = tmp_path / "results" / "uncertainty"
    unc_dir.mkdir(parents=True)
    pd.DataFrame({"image_id": ids, "uncertainty": np.linspace(0, 1, len(ids))}).to_parquet(
        unc_dir / "base_s0__pool.parquet", index=False)
    angles = np.linspace(0, 1, len(ids))
    vectors = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype("float32")
    flat = faiss.IndexFlatIP(2)
    flat.add(vectors)
    lookup = pd.DataFrame({"row_id": range(len(ids)), "image_id": ids, "tile_idx": 0})
    index = Index(index_root, {"model_id": "siglip384", "backend": "openvino_gpu", "n_tiles": 1,
                              "n_vectors": len(ids), "open_clip_name": "test", "pretrained": "test",
                              "preprocess": "pad_square"},
                  flat, lookup)

    fail_text = [False]
    class Clip:
        model_id = "siglip384"
        backend = "torch_cpu"

        def encode_text(self, prompts):
            if fail_text[0]:
                fail_text[0] = False
                raise RuntimeError("test encoder failure")
            return np.tile(np.array([[1, 0]], dtype="float32"), (len(prompts), 1))

        def encode_images(self, images):
            return np.tile(np.array([[1, 0]], dtype="float32"), (len(images), 1))

    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(config, "load_paths", lambda: SimpleNamespace(
        data_root=tmp_path, meta_dir=meta_dir, index_root=index_root, clip_weights_dir=tmp_path,
        uncertainty_file=unc_dir / "base_s0__pool.parquet"))
    monkeypatch.setattr(Index, "load", lambda folder: index)
    encoders, backends = [], []
    def load(*args):
        clip = Clip()
        encoders.append(clip)
        return clip
    def compile_image(clip, backend, *args):
        backends.append(backend)
        clip.backend = backend
    monkeypatch.setattr(encode, "load_clip", load)
    monkeypatch.setattr(search_backends, "set_backend", compile_image)
    st.cache_resource.clear()
    st.cache_data.clear()
    app = AppTest.from_file(str(Path(__file__).resolve().parents[2] / "app" / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    app.button(key="go_txt").click().run()
    assert not app.exception
    initial = app.session_state["search_txt"]["rows"].copy()
    assert len(encoders) == 1
    assert backends == []  # Text-only search must never compile an image backend.
    app.run()  # A rerun previously discarded all search results.
    assert not app.exception
    pd.testing.assert_frame_equal(app.session_state["search_txt"]["rows"], initial)
    first_id = initial.iloc[0].image_id
    assert all(button.label != "Xem ảnh lớn" for button in app.button)
    app.button(key=f"like_txt_{first_id}").click().run()
    assert not app.exception
    related = app.session_state["search_txt"]
    assert related["query"] == "image:" + first_id
    assert first_id not in set(related["rows"].image_id)
    assert len(app.session_state["search_history_txt"]) == 1
    assert backends == ["openvino_gpu"]
    assert encoders[0].backend == "torch_cpu"  # The cached base encoder remains unmodified.
    app.button(key="search_back_txt").click().run()
    assert not app.exception
    pd.testing.assert_frame_equal(app.session_state["search_txt"]["rows"], initial)

    # The image tab has an independent result set and independent widget keys.
    app.text_input[0].set_value(ids[5]).run()
    app.button(key="go_img").click().run()
    assert not app.exception
    assert app.session_state["search_img"]["query"] == "image:" + ids[5]
    pd.testing.assert_frame_equal(app.session_state["search_txt"]["rows"], initial)
    image_first = app.session_state["search_img"]["rows"].iloc[0].image_id
    app.button(key=f"like_img_{image_first}").click().run()
    assert not app.exception
    assert image_first not in set(app.session_state["search_img"]["rows"].image_id)
    pd.testing.assert_frame_equal(app.session_state["search_txt"]["rows"], initial)

    # A changed filter cannot silently reuse results or export the old context.
    app.multiselect[0].select("rainy").run()
    assert not app.exception
    assert any("Thiết lập tìm kiếm đã đổi" in entry.value for entry in app.info)
    app.button(key="go_txt").click().run()
    assert not app.exception
    assert set(app.session_state["search_txt"]["rows"].weather) == {"rainy"}

    # A request arriving while one is pending cannot replace it or start a second job.
    generation = app.session_state["search_generation"]
    app.session_state["search_request"] = ("text", "txt", None)
    app.button(key="go_img").click().run()
    assert not app.exception
    assert app.session_state["search_generation"] == generation
    assert app.session_state.get("search_request") is None

    # Failure preserves previous results and releases both buttons so the user can retry.
    before_error = app.session_state["search_txt"]["rows"].copy()
    fail_text[0] = True
    app.button(key="go_txt").click().run()
    assert not app.exception
    assert any("test encoder failure" in message.value for message in app.warning)
    assert app.session_state.get("search_request") is None
    assert not app.button(key="go_txt").disabled
    assert not app.button(key="go_img").disabled
    pd.testing.assert_frame_equal(app.session_state["search_txt"]["rows"], before_error)
    app.button(key="go_txt").click().run()
    assert not app.exception
    assert len(encoders) == 1
    assert backends == ["openvino_gpu"]

    assert [tab.label for tab in app.tabs] == ["Tìm bằng câu", "Tìm bằng ảnh", "So sánh"]
    before_compare = app.session_state["search_txt"]["rows"].copy()
    app.button(key="go_cmp").click().run()
    assert not app.exception
    comparison = app.session_state["search_cmp"]["rows"]
    assert set(comparison) == {"whole", "t3x2o20"}
    pd.testing.assert_frame_equal(app.session_state["search_txt"]["rows"], before_compare)
