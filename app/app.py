"""N3-07B demo app (Streamlit, offline on the laptop).

Gear chain per search:
  sentence(s) or example image -> CLIP encoder of the SAME model as the chosen index -> query vector(s)
  -> FAISS: nearest crop vectors -> lookup: row -> (image_id, tile) -> image score = max over tiles/prompts
  -> metadata filters -> optional sort by detector uncertainty -> thumbnails + CSV export.
CLIP only encodes; FAISS only finds nearest vectors; the lookup table maps rows back to images.

Run: streamlit run app/app.py
"""
from __future__ import annotations

import io
import sys
import time
from copy import copy
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import CONFIG_DIR, REPO_ROOT, load_paths, load_yaml  # noqa: E402
from src.search.backends import set_backend  # noqa: E402
from src.search.encode import load_clip  # noqa: E402
from src.search.search import Index, search_vectors  # noqa: E402
from src.search.tiling import tile_boxes  # noqa: E402

DEFAULT_N = 50
st.set_page_config(page_title="N3-07B rare-case search", layout="wide")
try:
    PATHS = load_paths()
except FileNotFoundError:
    st.title("N3-07B · Cài đặt dữ liệu cục bộ")
    st.info("Sao chép configs/paths.example.yaml thành configs/paths.yaml, rồi trỏ đến ảnh, metadata và chỉ mục của bạn.")
    st.code("python tools/check_assets.py")
    st.caption("Bản clone chỉ chứa mã chạy web; dữ liệu ảnh, chỉ mục và trọng số được chuẩn bị riêng theo README.")
    st.stop()
TILINGS = load_yaml(CONFIG_DIR / "tiling.yaml")["tilings"]
META_LABELS = {
    "rainy": "Mưa", "clear": "Quang", "snowy": "Tuyết", "overcast": "Âm u",
    "partly cloudy": "Ít mây", "foggy": "Sương mù", "undefined": "Chưa xác định",
    "city street": "Đường phố", "highway": "Cao tốc", "residential": "Khu dân cư",
    "parking lot": "Bãi đỗ xe", "tunnel": "Đường hầm", "gas stations": "Trạm xăng",
    "daytime": "Ban ngày", "night": "Ban đêm", "dawn/dusk": "Bình minh / hoàng hôn",
}


def meta_label(value: str) -> str:
    return META_LABELS.get(value, value)


# ---------------- cached resources ----------------
@st.cache_resource(show_spinner="Nạp chỉ mục FAISS...")
def get_index(index_id: str) -> Index:
    return Index.load(PATHS.index_root / index_id)


@st.cache_resource(show_spinner=False)
def get_clip(model_id: str):
    return load_clip(model_id, PATHS.clip_weights_dir)


@st.cache_resource(show_spinner=False)
def get_image_clip(model_id: str, image_backend: str):
    # Share immutable torch weights, but keep each compiled image backend separate.
    clip = copy(get_clip(model_id))
    if image_backend.startswith("openvino"):
        try:
            set_backend(clip, image_backend, PATHS.index_root / "_ov_cache", 28)
        except Exception as e:                       # keep text search alive; image query falls back
            st.warning(f"Không mở được {image_backend} ({e}); ảnh truy vấn dùng torch_cpu")
    return clip


@st.cache_data(show_spinner=False)
def get_images() -> pd.DataFrame:
    return pd.read_parquet(PATHS.meta_dir / "images.parquet").set_index("image_id")


@st.cache_data(show_spinner=False)
def get_uncertainty() -> pd.DataFrame | None:
    f = getattr(PATHS, "uncertainty_file", None)
    return pd.read_parquet(f).set_index("image_id") if f and f.exists() else None


def available_indexes() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for d in sorted(PATHS.index_root.glob("*__*")):
        if "_dbg" in d.name or not (d / "index.faiss").exists():
            continue
        model, tiling = d.name.split("__", 1)
        out.setdefault(model, []).append(tiling)
    return out


# ---------------- search helpers ----------------
def run_search(index_id: str, prompts: list[str] | None, image: Image.Image | None,
               n: int, filters: dict, progress=None) -> tuple[pd.DataFrame, float]:
    t = time.perf_counter()
    report = progress or (lambda value, text: None)
    report(0, "1/4 · Đang chuẩn bị bộ tìm kiếm…")
    idx = get_index(index_id)
    clip = (get_clip(idx.meta["model_id"]) if prompts else
            get_image_clip(idx.meta["model_id"], idx.meta["backend"]))
    need_full = any(filters.values())
    top = None if need_full else n
    idx.check_model(clip.model_id)
    report(25, "2/4 · Đang xử lý câu tìm…" if prompts else "2/4 · Đang xử lý ảnh mẫu…")
    if prompts:
        vectors = clip.encode_text(prompts)
        query_ids = [f"p{i + 1}" for i in range(len(prompts))]
    else:
        vectors = clip.encode_images([image])  # OI-17: whole image
        query_ids = ["image"]
    report(50, "3/4 · Đang tìm ảnh phù hợp…")
    r = search_vectors(idx, vectors, top_n=top, query_ids=query_ids)
    report(75, "4/4 · Đang lọc và sắp xếp kết quả…")
    meta = get_images()
    r = r.join(meta[["weather", "scene", "timeofday", "image_path"]], on="image_id")
    for col, allowed in filters.items():
        if allowed:
            r = r[r[col].isin(allowed)]
    unc = get_uncertainty()
    r["uncertainty"] = unc["uncertainty"].reindex(r["image_id"]).to_numpy() if unc is not None else np.nan
    report(100, "Đã tìm xong · Đang hiển thị ảnh…")
    return r.head(n).reset_index(drop=True), time.perf_counter() - t


@st.cache_data(show_spinner=False, max_entries=200)
def thumb(image_path: str, best_tile: int, tiling_id: str) -> bytes | str:
    """Full-resolution grid image, optionally marking the best matching tile.
    Without a tile box the jpg file is sent as is (no re-encoding)."""
    if not (best_tile > 0 and tiling_id in TILINGS):
        return str(PATHS.data_root / image_path)
    with Image.open(PATHS.data_root / image_path) as source:
        img = source.convert("RGB")
    x1, y1, x2, y2 = tile_boxes(img.width, img.height, TILINGS[tiling_id])[best_tile]
    ImageDraw.Draw(img).rectangle([x1, y1, x2 - 1, y2 - 1], outline=(255, 200, 0), width=6)
    encoded = io.BytesIO()
    img.save(encoded, format="JPEG", quality=92)
    return encoded.getvalue()


def show_image(image, key: str, **kwargs) -> None:
    """Every tab uses the same client-side click preview, with no search or label information."""
    with st.container(key=f"preview_image_{key}"):
        st.image(image, **kwargs)


def show_grid(r: pd.DataFrame, tiling_id: str, cols: int = 3, mark: set | None = None,
              search_slot: str | None = None, viewer_group: str | None = None) -> None:
    for start in range(0, len(r), cols):
        cs = st.columns(cols)
        for offset, (c, row) in enumerate(zip(cs, r.iloc[start:start + cols].itertuples())):
            tag = " ★" if mark and row.image_id in mark else ""
            unc = "" if np.isnan(row.uncertainty) else f" · u={row.uncertainty:.2f}"
            group = search_slot or viewer_group or tiling_id
            with c.container(key=f"gallery_card_{group}_{row.image_id}"):
                show_image(thumb(row.image_path, int(row.best_tile_idx), tiling_id),
                           f"{viewer_group or search_slot or tiling_id}_{row.image_id}", width="stretch")
                st.html(f'<div class="n3-card-meta"><span class="n3-rank">#{start + offset + 1}{tag}</span>'
                        f'<span>{escape(meta_label(row.weather))}</span>'
                        f'<span>{escape(meta_label(row.timeofday))}</span></div>')
                if search_slot:
                    st.button("Tìm ảnh giống", icon=":material/image_search:", use_container_width=True,
                              key=f"like_{search_slot}_{row.image_id}", on_click=find_similar,
                              args=(search_slot, row.image_id), disabled=busy)
                elif viewer_group and viewer_group.startswith("cmp_"):
                    st.button("Tìm ảnh giống", icon=":material/image_search:", use_container_width=True,
                              key=f"action_cmp_like_{viewer_group}_{row.image_id}", on_click=request_search,
                              args=("compare_similar", "cmp", row.image_id), disabled=busy)
                with st.expander("Chi tiết"):
                    st.caption(f"ID: {row.image_id}\nĐiểm tương đồng: {row.score:.3f}\n"
                               f"Ô cắt: {row.best_tile_idx}{unc}\n"
                               f"Bối cảnh: {meta_label(row.scene)}")


def search_context(index_id: str, n: int, filters: dict, sort_unc: bool) -> tuple:
    return index_id, int(n), tuple((k, tuple(sorted(v))) for k, v in sorted(filters.items())), bool(sort_unc)


def save_search(slot: str, r: pd.DataFrame, dt: float, query: str, label: str,
                context: tuple, preview: Image.Image | str | None = None,
                keep_history: bool = False) -> None:
    if context[3]:
        r = r.sort_values("uncertainty", ascending=False, kind="mergesort")
    store_result(slot, {"rows": r.reset_index(drop=True), "seconds": dt, "query": query,
                        "label": label, "context": context, "preview": preview}, keep_history)


def store_result(slot: str, result: dict, keep_history: bool = False) -> None:
    key, history_key = f"search_{slot}", f"search_history_{slot}"
    if keep_history and key in st.session_state:
        history = st.session_state.get(history_key, [])
        st.session_state[history_key] = (history + [st.session_state[key]])[-10:]
    else:
        st.session_state[history_key] = []
    st.session_state[key] = result
    st.session_state.pop(f"search_error_{slot}", None)


def request_search(kind: str, slot: str, payload=None) -> None:
    # Callbacks only queue work: render disabled controls before doing anything expensive.
    if st.session_state.get("search_request") is None:
        st.session_state["search_request"] = (kind, slot, payload)
        st.session_state["search_generation"] = st.session_state.get("search_generation", 0) + 1


def operation_message(kind: str) -> str:
    if kind == "back":
        return "Đang khôi phục kết quả…"
    if kind.startswith("compare"):
        return "Đang so sánh hai cấu hình…"
    return "Đang tìm ảnh… Vui lòng chờ."


def find_similar(slot: str, image_id: str) -> None:
    request_search("similar", slot, image_id)


def execute_search(context: tuple, prompts: list[str], upload, input_id: str, placeholders: dict) -> None:
    kind, slot, image_id = st.session_state["search_request"]
    loading = placeholders[slot].progress(0, text=operation_message(kind))
    try:
        if slot == "cmp":
            execute_comparison(kind, image_id, context, loading)
            return
        if kind == "back":
            previous_search(slot)
            return
        index_id, count, filter_items, _ = context
        image = None
        preview = None
        if kind == "text":
            if not prompts:
                raise ValueError("Nhập ít nhất một câu trước khi tìm.")
            query = label = " | ".join(prompts)
        else:
            if kind == "similar":
                current = st.session_state[f"search_{slot}"]
                if current["context"] != context:
                    raise ValueError("Thiết lập đã đổi. Bấm Tìm để cập nhật kết quả.")
                preview = str(PATHS.data_root / get_images().at[image_id, "image_path"])
                with Image.open(preview) as source:
                    image = source.convert("RGB")
                source_name = image_id
            elif upload is not None:
                with Image.open(io.BytesIO(upload.getvalue())) as source:
                    image = source.convert("RGB")
                source_name = upload.name
                preview = image
            elif input_id.strip() in get_images().index:
                source_name = input_id.strip()
                preview = str(PATHS.data_root / get_images().at[source_name, "image_path"])
                with Image.open(preview) as source:
                    image = source.convert("RGB")
            else:
                raise ValueError("Chưa có ảnh hoặc image_id không tồn tại.")
            query, label = f"image:{source_name}", f"Ảnh giống {source_name}"
        r, dt = run_search(index_id, prompts if kind == "text" else None, image,
                           count + (kind == "similar"), dict(filter_items), progress=loading.progress)
        if kind == "similar":
            r = r[r["image_id"] != image_id].head(count)
        save_search(slot, r, dt, query, label, context, preview=preview, keep_history=kind == "similar")
    except Exception as e:
        st.session_state[f"search_error_{slot}"] = f"Chưa hoàn tất: {e}"
    finally:
        st.session_state.pop("search_request", None)
        st.rerun()


def previous_search(slot: str) -> None:
    history = st.session_state.get(f"search_history_{slot}", [])
    if history:
        st.session_state[f"search_{slot}"] = history[-1]
        st.session_state[f"search_history_{slot}"] = history[:-1]
    st.session_state.pop(f"search_error_{slot}", None)


def show_search_results(slot: str, context: tuple) -> None:
    if error := st.session_state.pop(f"search_error_{slot}", None):
        st.warning(error)
    if st.session_state.get("search_request") is not None:
        return
    current = st.session_state.get(f"search_{slot}")
    if current is None:
        st.html('<div class="n3-empty">Kết quả sẽ hiển thị ở đây sau khi tìm kiếm.</div>')
        return
    if current["context"] != context:
        st.info("Thiết lập tìm kiếm đã đổi. Bấm Tìm để cập nhật kết quả.")
        return
    r = current["rows"]
    index_id = context[0]
    with st.container(key=f"result_toolbar_{slot}"):
        summary, download = st.columns([5, 1], vertical_alignment="center")
        with summary:
            st.markdown(f"**{len(r)} ảnh** · {current['seconds']:.2f} s")
            st.caption(current["label"])
        with download:
            st.download_button("Xuất CSV", csv_bytes(r, index_id, current["query"]),
                               f"selection_{index_id}_{slot}.csv", "text/csv", key=f"download_{slot}",
                               on_click="ignore", use_container_width=True)
    if st.session_state.get(f"search_history_{slot}"):
        st.button("Quay lại kết quả trước", icon=":material/arrow_back:", key=f"search_back_{slot}",
                  on_click=request_search, args=("back", slot), disabled=busy)
    if current["preview"] is not None:
        with st.expander("Ảnh mẫu đang tìm"):
            show_image(current["preview"], f"{slot}_query", width=280, caption="Ảnh mẫu đang tìm")
    if r.empty:
        st.info("Không có ảnh phù hợp với thiết lập hiện tại.")
        return
    show_grid(r, index_id.split("__", 1)[1], search_slot=slot)


def execute_comparison(kind: str, image_id, context: tuple, loading) -> None:
    if kind == "back":
        previous_search("cmp")
        return
    model = context[0].split("__", 1)[0]
    count, filter_items, sort_unc = context[1:]
    image = preview = None
    prompts = [st.session_state["cmp_q"].strip()]
    query = label = prompts[0]
    similar = kind == "compare_similar"
    if similar:
        if st.session_state["search_cmp"]["context"] != context:
            raise ValueError("Thiết lập đã đổi. Bấm So sánh để cập nhật kết quả.")
        preview = str(PATHS.data_root / get_images().at[image_id, "image_path"])
        with Image.open(preview) as source:
            image = source.convert("RGB")
        prompts = None
        query, label = f"image:{image_id}", f"Ảnh giống {image_id}"
    elif not query:
        raise ValueError("Nhập câu truy vấn trước khi so sánh.")
    rows, seconds = {}, 0.0
    for position, tiling in enumerate(("whole", "t3x2o20")):
        def report(value, text, position=position):
            loading.progress(int((position * 100 + value) / 2), text=f"{position + 1}/2 cấu hình · {text}")
        r, dt = run_search(f"{model}__{tiling}", prompts, image, count + similar, dict(filter_items), report)
        if similar:
            r = r[r.image_id != image_id].head(count)
        if sort_unc:
            r = r.sort_values("uncertainty", ascending=False, kind="mergesort")
        rows[tiling] = r.reset_index(drop=True)
        seconds += dt
    store_result("cmp", {"rows": rows, "query": query, "label": label, "seconds": seconds,
                         "context": context, "preview": preview}, keep_history=similar)


def show_comparison(context: tuple) -> None:
    if error := st.session_state.pop("search_error_cmp", None):
        st.warning(error)
    current = st.session_state.get("search_cmp")
    if not current or busy:
        return
    if current["context"] != context:
        st.info("Thiết lập đã đổi. Bấm So sánh để cập nhật kết quả.")
        return
    st.caption(f"{current['label']} · {current['seconds']:.2f} s")
    if st.session_state.get("search_history_cmp"):
        st.button("Quay lại kết quả trước", key="action_cmp_back", icon=":material/arrow_back:",
                  on_click=request_search, args=("back", "cmp"), disabled=busy)
    if current["preview"] is not None:
        show_image(current["preview"], "cmp_query", width=280, caption="Ảnh mẫu đang tìm")
    a, b = current["rows"]["whole"], current["rows"]["t3x2o20"]
    common = set(a.image_id) & set(b.image_id)
    st.caption(f"{len(common)} ảnh chung (★) trong top-{context[1]}")
    for column, tiling, title in zip(st.columns(2), ("whole", "t3x2o20"),
                                     ("Ảnh nguyên", "Ảnh nguyên + 6 ô (khung vàng = ô khớp nhất)")):
        with column:
            st.subheader(title)
            index_id = f"{context[0].split('__', 1)[0]}__{tiling}"
            r = current["rows"][tiling]
            st.download_button("Xuất CSV", csv_bytes(r, index_id, current["query"]),
                               f"comparison_{index_id}.csv", "text/csv", key=f"download_cmp_{tiling}",
                               on_click="ignore")
            if r.empty:
                st.info("Không có ảnh phù hợp với thiết lập hiện tại.")
            show_grid(r, tiling, cols=1, mark=common, viewer_group=f"cmp_{tiling}")


def csv_bytes(r: pd.DataFrame, index_id: str, query: str) -> bytes:
    out = r.assign(index_id=index_id, query=query)[
        ["image_id", "index_id", "query", "score", "best_tile_idx", "uncertainty", "weather", "timeofday"]]
    return out.to_csv(index=False).encode("utf-8")


# ---------------- UI ----------------
# Only trusted, repository-owned code is executed; image URLs come from the rendered grid.
request = st.session_state.get("search_request")
busy = request is not None
blocked = busy
generation = st.session_state.get("search_generation", 0)
st.html(Path(__file__).with_name("gallery_theme.html"), unsafe_allow_javascript=True)
st.html(Path(__file__).with_name("image_preview.html"), unsafe_allow_javascript=True)
st.html(Path(__file__).with_name("search_feedback.html"), unsafe_allow_javascript=True)
message = operation_message(request[0]) if request else ""
st.html(f'<span hidden id="n3-search-active" data-generation="{generation}" '
        f'data-busy="{str(busy).lower()}" data-message="{message}"></span>')
st.html('<header class="n3-header"><div class="n3-brand"><span class="n3-logo">N3</span>'
        '<div><h1>N3-07B</h1><p>Khám phá ca hiếm trong dữ liệu giao thông</p></div></div>'
        '<button type="button" id="n3-theme-toggle" aria-label="Đổi giao diện sáng tối">'
        'Đổi sáng / tối</button></header>')
idxs = available_indexes()
if not idxs:
    st.error(f"Không có chỉ mục nào trong {PATHS.index_root}")
    st.stop()

if not (PATHS.meta_dir / "images.parquet").exists():
    st.warning("Chưa có images.parquet ở meta_dir. Kiểm tra configs/paths.yaml theo hướng dẫn README.")
    st.stop()
meta = get_images()
with st.container(key="gallery_filters"):
    filter_columns = st.columns(3)
    filters = {}
    for column, name, label in zip(filter_columns, ("weather", "scene", "timeofday"),
                                    ("Thời tiết", "Bối cảnh", "Thời điểm")):
        with column:
            filters[name] = st.multiselect(label, sorted(meta[name].unique()), format_func=meta_label,
                                            placeholder="Tất cả", key=f"filter_{name}", disabled=blocked)

with st.expander("Thiết lập nâng cao", icon=":material/tune:"):
    model_col, tiling_col, count_col = st.columns(3)
    model = model_col.selectbox("Model CLIP", list(idxs),
                                index=list(idxs).index("siglip384") if "siglip384" in idxs else 0,
                                disabled=blocked)
    tiling = tiling_col.selectbox("Cách cắt", idxs[model],
                                  index=idxs[model].index("t3x2o20") if "t3x2o20" in idxs[model] else 0,
                                  disabled=blocked)
    index_id = f"{model}__{tiling}"
    m = get_index(index_id).meta
    st.caption(f"**{index_id}** · {m['open_clip_name']} / {m['pretrained']} · preprocess {m['preprocess']} · "
               f"backend {m['backend']} · {m['n_vectors']:,} vectors")
    n = count_col.number_input("Số ảnh hiển thị", 10, 200, DEFAULT_N, step=10, disabled=blocked)
    sort_unc = st.checkbox("Sắp xếp theo độ bất định (cao → thấp)", disabled=blocked or get_uncertainty() is None,
                           help="Tuỳ chọn: cấu hình uncertainty_file trong configs/paths.yaml")

context = search_context(index_id, n, filters, sort_unc)

tab_txt, tab_img, tab_cmp = st.tabs(["Tìm bằng câu", "Tìm bằng ảnh", "So sánh"])

with tab_txt:
    with st.container(key="query_txt", border=True):
        query_col, submit_col = st.columns([6, 1], vertical_alignment="bottom")
        txt = query_col.text_area("1–5 câu tiếng Anh, mỗi dòng một câu",
                                  "a pedestrian crossing the street at night in the rain", disabled=blocked, height=76)
        submit_col.button("Tìm", key="go_txt", disabled=blocked, on_click=request_search, args=("text", "txt"),
                           type="primary", icon=":material/search:", use_container_width=True)
    prompts = [p.strip() for p in txt.splitlines() if p.strip()][:5]
    txt_loading = st.empty()
    show_search_results("txt", context)

with tab_img:
    with st.container(key="query_img", border=True):
        upload_col, id_col = st.columns([3, 2], vertical_alignment="bottom")
        up = upload_col.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png"], disabled=blocked)
        iid = id_col.text_input("Hoặc nhập mã ảnh (image_id)", disabled=blocked)
        st.button("Tìm ảnh tương tự", key="go_img", disabled=blocked, on_click=request_search, args=("image", "img"),
                  type="primary", icon=":material/search:")
    img_loading = st.empty()
    show_search_results("img", context)

with tab_cmp:
    if not {"whole", "t3x2o20"} <= set(idxs[model]):
        st.info("Cần cả hai chỉ mục whole và t3x2o20 của model này")
    else:
        with st.container(key="query_cmp", border=True):
            query_col, submit_col = st.columns([6, 1], vertical_alignment="bottom")
            query_col.text_input("Câu truy vấn", "a bicycle at night", key="cmp_q", disabled=blocked)
            submit_col.button("So sánh", key="go_cmp", on_click=request_search, args=("compare", "cmp"),
                               disabled=blocked, type="primary", use_container_width=True)
    cmp_loading = st.empty()
    show_comparison(context)

if busy:
    execute_search(context, prompts, up, iid, {"txt": txt_loading, "img": img_loading, "cmp": cmp_loading})
# Emitted only after the new result grid is rendered (including validation/failure paths).
st.html(f'<span hidden id="n3-search-finished" data-generation="{generation}"></span>')
