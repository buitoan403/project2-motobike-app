# project2_motobike_app.py
# -*- coding: utf-8 -*-
"""
Project 2: Đề xuất xe máy dựa trên nội dung, phân cụm xe máy
Chạy trên Visual Studio Code / Streamlit.
"""

import os
import re
import pickle
from math import ceil
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt


# ======================================================
# CÁC HẰNG SỐ ĐƯỜNG DẪN
# ======================================================

CLUSTER_DATA_FILE = "data_motobikes_clean.xlsx"  # dữ liệu cho phân cụm
SEARCH_DATA_FILE = "data_motobikes.xlsx"         # dữ liệu cho hệ gợi ý
COSINE_PKL_FILE = "cosine_sim_model.pkl"         # ma trận cosine
HEADER_IMAGE_FILE = "xe.png"                     # ảnh header chung (nếu có)
SEARCH_HEADER_IMAGE_FILE = "b12bca47-fea2-499d-80f1-1915896b8525.png"  # ảnh trang tìm kiếm (nếu có)


# ======================================================
# HÀM TIỆN ÍCH CHUNG
# ======================================================

def get_file_path(filename: str) -> str:
    """Trả về đường dẫn tuyệt đối tới file nằm cùng thư mục với script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, filename)


def get_image_path(filename: str):
    """Trả về đường dẫn ảnh nếu tồn tại, ngược lại trả về None."""
    path = get_file_path(filename)
    if os.path.exists(path):
        return path
    return None


# ======================================================
# PHẦN 1: PHÂN CỤM XE MÁY (KMEANS + PCA + FORM DỰ ĐOÁN)
# ======================================================

def parse_price_to_million(s: str):
    """Chuẩn hóa chuỗi giá về đơn vị triệu đồng."""
    if pd.isna(s):
        return np.nan
    s = str(s).lower()

    # loại bỏ ký tự không cần thiết
    s = s.replace("\u00a0", " ")
    s = s.replace("vnđ", "").replace("vnd", "").replace("đ", "")
    s = s.replace(",", ".").strip()

    m = re.search(r"(\d+\.?\d*)", s)
    if not m:
        return np.nan

    num = float(m.group(1))

    if "triệu" in s or " tr" in s:
        return num
    if "nghìn" in s or "ngàn" in s or "k" in s:
        return num / 1000
    # nếu chỉ ghi dạng 20.000.000
    if num > 1000:
        return num / 1_000_000
    return num


@st.cache_data
def load_and_prepare_cluster_data(data_path: str):
    """
    Đọc & xử lý dữ liệu phân cụm,
    trả về df, numeric_cols, categorical_cols, preprocess, X_dense.
    """
    ext = os.path.splitext(data_path)[1].lower()
    if ext == ".csv":
        df_raw = pd.read_csv(data_path)
    else:
        df_raw = pd.read_excel(data_path)

    df = df_raw.copy()

    # Tự tìm cột khoảng giá min / max
    min_col_txt = [c for c in df.columns if "min" in c.lower()][0]
    max_col_txt = [c for c in df.columns if "max" in c.lower()][0]

    df["Khoảng giá min (triệu)"] = df[min_col_txt].apply(parse_price_to_million)
    df["Khoảng giá max (triệu)"] = df[max_col_txt].apply(parse_price_to_million)

    # Cột Giá chính
    if "Giá" in df.columns:
        df["Giá"] = pd.to_numeric(df["Giá"], errors="coerce")
        mask = df["Giá"].isna()
        df.loc[mask, "Giá"] = df.loc[
            mask, ["Khoảng giá min (triệu)", "Khoảng giá max (triệu)"]
        ].mean(axis=1)
    else:
        df["Giá"] = df[["Khoảng giá min (triệu)", "Khoảng giá max (triệu)"]].mean(axis=1)

    # Tuổi xe
    df["Năm đăng ký"] = pd.to_numeric(df["Năm đăng ký"], errors="coerce")
    df["Tuổi xe"] = 2025 - df["Năm đăng ký"]

    # Km
    if "Số Km đã đi" in df.columns:
        df["Số Km đã đi"] = pd.to_numeric(df["Số Km đã đi"], errors="coerce")
    else:
        df["Số Km đã đi"] = np.nan

    # Các cột dùng phân cụm
    numeric_cols = ["Giá", "Tuổi xe", "Số Km đã đi"]
    categorical_cols = [
        "Thương hiệu",
        "Dòng xe",
        "Loại xe",
        "Dung tích xe",
        "Xuất xứ",
    ]

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    X = df[numeric_cols + categorical_cols].copy()

    pre_num = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    pre_cat = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess = ColumnTransformer(
        [
            ("num", pre_num, numeric_cols),
            ("cat", pre_cat, categorical_cols),
        ]
    )

    X_prep = preprocess.fit_transform(X)
    X_dense = X_prep.toarray()

    return df, numeric_cols, categorical_cols, preprocess, X_dense


def run_kmeans(df, numeric_cols, X_dense, K: int):
    """Chạy KMeans, trả về model + kết quả."""
    model = KMeans(n_clusters=K, n_init=10, random_state=42)
    labels = model.fit_predict(X_dense)

    sil = silhouette_score(X_dense, labels)

    dfc = df.copy()
    dfc["cluster"] = labels

    summary = (
        dfc.groupby("cluster")[numeric_cols]
        .agg(["count", "mean", "min", "max"])
        .round(2)
    )

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_dense)

    return {
        "model": model,
        "dfc": dfc,
        "summary": summary,
        "silhouette": sil,
        "X_pca": X_pca,
        "K": K,
    }


def seg_label(c: int) -> str:
    return f"Phân khúc {c + 1}"


def render_header():
    """Tiêu đề + ảnh xe ở góc phải."""
    img_path = get_image_path(HEADER_IMAGE_FILE)

    if img_path:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.title("Project 2 – Đề xuất & phân khúc xe máy đã qua sử dụng")
        with col2:
            st.image(img_path, use_column_width=True)
    else:
        st.title("Project 2 – Đề xuất & phân khúc xe máy đã qua sử dụng")


def page_project_intro():
    st.header("Giới thiệu dự án")
    st.write(
        """
**Project 2: Đề xuất xe máy dựa trên nội dung, phân cụm xe máy**

Mục tiêu:
- Xây dựng hệ thống **phân khúc xe máy đã qua sử dụng** bằng KMeans & PCA;
- Xây dựng hệ thống **đề xuất xe tương tự** dựa trên nội dung (content-based) với ma trận **cosine similarity**;
- Ứng dụng kết quả vào hỗ trợ quyết định cho **người mua** và **bên bán** (giá, phân khúc, lựa chọn sản phẩm).

Các chức năng chính:
1. **Phân khúc xe máy (KMeans)** – trực quan hóa PCA, xem thống kê từng phân khúc & dự đoán phân khúc cho xe mới.
2. **Tìm kiếm & gợi ý xe tương tự** – tìm xe theo id/từ khóa, gợi ý danh sách xe giống nhau về nội dung.
"""
    )


def page_evaluation(result):
    st.header("Đánh giá & Báo cáo (KMeans)")

    st.subheader("1️⃣ Thông tin mô hình")
    st.write(f"- Số phân khúc (K): **{result['K']}**")
    st.write(f"- Giá trị Silhouette: **{result['silhouette']:.4f}**")
    st.markdown(
        """
- Silhouette càng lớn (gần 1) → các phân khúc càng tách biệt, chất lượng phân cụm càng tốt.
"""
    )

    st.subheader("2️⃣ Thống kê theo từng phân khúc (chỉ tiêu numeric)")
    summary = result["summary"].copy()
    summary.index = [seg_label(i) for i in summary.index]
    st.dataframe(summary, use_container_width=True)


def page_cluster_and_predict(df, numeric_cols, categorical_cols, preprocess, result):
    """
    Trang: Khám phá & Dự đoán phân khúc
    - Hiển thị PCA, bảng xe theo phân khúc
    - Form dự đoán phân khúc cho xe mới
    - SAU KHI DỰ ĐOÁN: Hiện thông tin chi tiết về phân khúc được dự đoán
    """
    st.header("Khám phá & Dự đoán phân khúc (KMeans + PCA)")

    dfc = result["dfc"]
    X_pca = result["X_pca"]
    model = result["model"]

    # ----- PCA plot
    st.subheader("🌈 Trực quan PCA 2D theo phân khúc")
    fig, ax = plt.subplots(figsize=(8, 5))
    clusters = sorted(dfc["cluster"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(clusters)))

    for cl, color in zip(clusters, colors):
        mask = dfc["cluster"] == cl
        ax.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=10,
            color=color,
            label=seg_label(cl),
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()
    st.pyplot(fig)

    # ----- Bảng chi tiết từng phân khúc
    st.subheader("📄 Danh sách xe theo phân khúc")
    choice = st.selectbox(
        "Chọn phân khúc muốn xem:",
        clusters,
        format_func=seg_label,
    )
    st.dataframe(
        dfc[dfc["cluster"] == choice].reset_index(drop=True),
        use_container_width=True,
    )

    st.markdown("---")

    # ----- Form dự đoán phân khúc cho xe người dùng
    st.subheader("🛵 Dự đoán phân khúc cho xe của bạn")

    defaults = {c: float(df[c].median()) for c in numeric_cols}
    cats = {c: sorted(df[c].dropna().unique()) for c in categorical_cols}

    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)

        thuong_hieu = col1.selectbox("Thương hiệu", cats.get("Thương hiệu", [""]))
        dong_xe = col2.selectbox("Dòng xe", cats.get("Dòng xe", [""]))
        loai_xe = col3.selectbox("Loại xe", cats.get("Loại xe", [""]))

        col4, col5, col6 = st.columns(3)
        dung_tich = col4.selectbox("Dung tích xe", cats.get("Dung tích xe", [""]))
        xuat_xu = col5.selectbox("Xuất xứ", cats.get("Xuất xứ", [""]))
        gia = col6.number_input(
            "Giá (triệu đồng)", value=defaults.get("Giá", 20.0), min_value=0.0
        )

        col7, col8 = st.columns(2)
        nam_dk = col7.number_input(
            "Năm đăng ký",
            min_value=1990,
            max_value=2025,
            value=int(df["Năm đăng ký"].median()),
        )
        so_km = col8.number_input(
            "Số Km đã đi",
            value=defaults.get("Số Km đã đi", 30000.0),
            min_value=0.0,
            step=1000.0,
        )

        submit = st.form_submit_button("🔍 Dự đoán phân khúc")

    if submit:
        tuoi_xe = 2025 - nam_dk

        row = {
            "Giá": gia,
            "Tuổi xe": tuoi_xe,
            "Số Km đã đi": so_km,
            "Thương hiệu": thuong_hieu,
            "Dòng xe": dong_xe,
            "Loại xe": loai_xe,
            "Dung tích xe": dung_tich,
            "Xuất xứ": xuat_xu,
        }

        X_user = preprocess.transform(pd.DataFrame([row])).toarray()
        pred = int(model.predict(X_user)[0])
        cluster_name = seg_label(pred)

        st.success(f"✅ Xe của bạn được xếp vào **{cluster_name}**.")

        # ==== THÔNG TIN CHI TIẾT VỀ PHÂN KHÚC DỰ ĐOÁN ====
        st.markdown("---")
        st.subheader(f"📊 Thông tin về {cluster_name}")

        dfc_cluster = dfc[dfc["cluster"] == pred].copy()
        cluster_size = len(dfc_cluster)

        st.write(f"- **Số lượng xe** trong {cluster_name}: **{cluster_size}** chiếc")

        # Thống kê các biến numeric trong phân khúc
        if numeric_cols:
            num_stats = (
                dfc_cluster[numeric_cols]
                .agg(["mean", "min", "max"])
                .round(2)
                .T
            )
            num_stats.columns = ["Trung bình", "Nhỏ nhất", "Lớn nhất"]
            st.write("**Thống kê các chỉ tiêu định lượng (trong phân khúc):**")
            st.dataframe(num_stats, use_container_width=True)

        # Thông tin phân bố một số biến categorical
        cat_info_lines = []
        for col in categorical_cols:
            if col in dfc_cluster.columns and dfc_cluster[col].notna().any():
                top_val = dfc_cluster[col].value_counts().idxmax()
                pct = (
                    dfc_cluster[col].value_counts(normalize=True).iloc[0] * 100
                )
                cat_info_lines.append(
                    f"- {col}: phổ biến nhất là **{top_val}** (~{pct:.1f}%)"
                )

        if cat_info_lines:
            st.write("**Đặc trưng định tính nổi bật trong phân khúc:**")
            st.markdown("\n".join(cat_info_lines))

        # Gợi ý: hiển thị vài xe tiêu biểu trong phân khúc
        st.write("**Một số xe tiêu biểu trong phân khúc:**")
        st.dataframe(
            dfc_cluster.head(10).reset_index(drop=True),
            use_container_width=True,
        )


def page_team():
    st.header("Thông tin nhóm thực hiện")
    st.write(
        """
**Nhóm học viên:**
1. Mai Bảo Ngọc  
2. Bùi Ngọc Toản  
3. Nguyễn Vũ Duy  
"""
    )


# ======================================================
# PHẦN 2: TÌM KIẾM & ĐỀ XUẤT XE TƯƠNG TỰ (CONTENT-BASED)
# ======================================================

@st.cache_resource(ttl=3600)
def load_search_data(path):
    """Đọc dữ liệu xe cho hệ gợi ý."""
    try:
        df = pd.read_excel(path, engine="openpyxl")
        df = df.reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Không thể đọc file dữ liệu: {path}\n{e}")
        return None


@st.cache_resource(ttl=3600)
def load_cosine_raw(path):
    """Thử đọc ma trận cosine similarity từ file .pkl (nếu có)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            cosine = pickle.load(f)
        return cosine
    except Exception as e:
        st.warning(f"Lỗi khi load ma trận cosine từ {path}: {e}")
        return None


@st.cache_resource(ttl=3600)
def build_cosine_from_df(df, pkl_path=None):
    """
    Xây dựng ma trận cosine similarity từ dữ liệu df
    dựa trên nội dung Tiêu đề + Mô tả chi tiết.
    Nếu pkl_path được cung cấp thì lưu lại ra file.
    """
    text_series = (
        df.get("Tiêu đề", "").fillna("").astype(str)
        + " "
        + df.get("Mô tả chi tiết", "").fillna("").astype(str)
    ).str.lower()

    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(text_series)

    cosine_sim = cosine_similarity(tfidf_matrix)

    if pkl_path is not None:
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(cosine_sim, f)
        except Exception as e:
            st.warning(f"Không thể lưu ma trận cosine ra file {pkl_path}: {e}")

    return cosine_sim


def get_or_create_cosine(df_bikes, cosine_path):
    """
    Lấy ma trận cosine nếu có, nếu không sẽ tự động build mới từ df_bikes.
    Đồng thời kiểm tra kích thước có khớp số dòng của df_bikes hay không.
    """
    cosine_sim = load_cosine_raw(cosine_path)
    if cosine_sim is None:
        st.info("Đang tạo mới ma trận cosine similarity từ dữ liệu...")
        cosine_sim = build_cosine_from_df(df_bikes, pkl_path=cosine_path)
    else:
        # nếu số dòng không khớp thì build lại
        try:
            if cosine_sim.shape[0] != len(df_bikes):
                st.info(
                    "Kích thước ma trận cosine không khớp số dòng dữ liệu. "
                    "Đang xây dựng lại ma trận cosine..."
                )
                cosine_sim = build_cosine_from_df(df_bikes, pkl_path=cosine_path)
        except Exception:
            st.info(
                "Không kiểm tra được kích thước ma trận cosine. "
                "Đang xây dựng lại ma trận cosine..."
            )
            cosine_sim = build_cosine_from_df(df_bikes, pkl_path=cosine_path)

    return cosine_sim


def find_best_title_match(df_titles, query):
    best_idx = None
    best_score = 0.0
    q = str(query).strip().lower()
    if not q:
        return None, 0.0
    for idx, title in enumerate(df_titles):
        t = str(title).lower()
        score = SequenceMatcher(None, q, t).ratio()
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx, best_score


def get_recommendations_by_index(df, cosine_sim, idx, top_k=30):
    if cosine_sim is None:
        return pd.DataFrame()
    try:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx]
        top_scores = sim_scores[:top_k]
        indices = [i for i, _ in top_scores]
        return df.iloc[indices].reset_index(drop=True)
    except Exception as e:
        st.error(f"Lỗi khi lấy gợi ý từ ma trận cosine: {e}")
        return pd.DataFrame()


def display_rows_with_expander(df_rows):
    if df_rows is None or df_rows.empty:
        st.write("_Không có kết quả để hiển thị._")
        return

    c0, c1, c2, c3, c4, c5 = st.columns([3, 2, 2, 1, 1, 1])
    c0.markdown("**Tiêu đề**")
    c1.markdown("**Thương hiệu**")
    c2.markdown("**Dòng xe**")
    c3.markdown("**Năm đăng ký**")
    c4.markdown("**Giá**")
    c5.markdown("**Chi tiết**")

    for _, row in df_rows.iterrows():
        t0, t1, t2, t3, t4, t5 = st.columns([3, 2, 2, 1, 1, 1])
        t0.write(row.get("Tiêu đề", ""))
        t1.write(row.get("Thương hiệu", ""))
        t2.write(row.get("Dòng xe", ""))
        t3.write(row.get("Năm đăng ký", ""))
        t4.write(row.get("Giá", ""))
        bike_id = row.get("id", "")
        label = f"Chi tiết ({bike_id})"
        with t5:
            with st.expander(label):
                desc = row.get("Mô tả chi tiết", "")
                if desc:
                    st.write(desc)
                else:
                    st.write("_Không có mô tả chi tiết._")


def paginate_dataframe(df, page, per_page):
    if df is None:
        return pd.DataFrame()
    start = (page - 1) * per_page
    end = start + per_page
    return df.iloc[start:end].reset_index(drop=True)


def page_search_and_recommend():
    """Trang Tìm kiếm & đề xuất xe tương tự."""
    img_path = get_image_path(SEARCH_HEADER_IMAGE_FILE)
    if img_path:
        try:
            st.image(img_path, use_column_width=True)
        except Exception:
            pass

    st.header("Tìm kiếm & Đề xuất xe máy tương tự (Content-based)")

    # load dữ liệu & cosine
    data_path = get_file_path(SEARCH_DATA_FILE)
    cosine_path = get_file_path(COSINE_PKL_FILE)

    df_bikes = load_search_data(data_path)
    if df_bikes is None:
        st.stop()

    cosine_sim = get_or_create_cosine(df_bikes, cosine_path)

    # session init (safe defaults)
    if "random_bikes" not in st.session_state:
        st.session_state.random_bikes = df_bikes.head(10).reset_index(drop=True)
    if "selected_bike_id" not in st.session_state:
        st.session_state.selected_bike_id = None
    if "page" not in st.session_state:
        st.session_state.page = 1
    if "max_results" not in st.session_state:
        st.session_state.max_results = 30
    if "per_page" not in st.session_state:
        st.session_state.per_page = 6
    if "last_query" not in st.session_state:
        st.session_state.last_query = ""
    if "last_query_method" not in st.session_state:
        st.session_state.last_query_method = ""

    # function callbacks
    def refresh_random_list():
        try:
            st.session_state.random_bikes = df_bikes.sample(n=10).reset_index(drop=True)
            st.session_state.selected_bike_id = None
            st.session_state.last_query = ""
            st.session_state.last_query_method = ""
            st.session_state.page = 1
            st.session_state.pop("selected_bike_option", None)
        except Exception as e:
            st.error("Lỗi khi làm mới danh sách: " + str(e))

    def on_select_change():
        val = st.session_state.get("selected_bike_option", None)
        if val:
            try:
                # val là tuple (title, id)
                st.session_state.selected_bike_id = val[1]
                st.session_state.last_query = str(st.session_state.selected_bike_id)
                st.session_state.last_query_method = "selectbox"
                st.session_state.page = 1
            except Exception:
                pass

    # --- Search UI: selection A and typed input B ---
    st.markdown("---")
    colA1, colA2 = st.columns([4, 1])
    with colA1:
        bike_options = [
            (row["Tiêu đề"], row["id"])
            for _, row in st.session_state.random_bikes.iterrows()
        ]
        st.selectbox(
            "Danh sách xe ngẫu nhiên",
            options=bike_options,
            format_func=lambda x: x[0] if isinstance(x, tuple) else str(x),
            key="selected_bike_option",
            on_change=on_select_change,
        )
    with colA2:
        if st.button("Làm mới danh sách"):
            refresh_random_list()

    q_input = st.text_input("Nhập id hoặc từ khóa tìm kiếm:", value="")

    # Thiết lập gợi ý
    st.markdown("**Thiết lập gợi ý**")
    cols_set = st.columns([1, 1, 2])
    with cols_set[0]:
        max_results = st.number_input(
            "Số gợi ý tối đa (tổng)",
            min_value=5,
            max_value=500,
            value=st.session_state.max_results,
            step=5,
            key="input_max_results",
        )
    with cols_set[1]:
        per_page = st.selectbox(
            "Số kết quả / trang",
            options=[3, 4, 6, 10],
            index=[
                3,
                4,
                6,
                10,
            ].index(st.session_state.per_page)
            if st.session_state.per_page in [3, 4, 6, 10]
            else 2,
            key="input_per_page",
        )

    # sync to session_state
    st.session_state.max_results = int(max_results)
    st.session_state.per_page = int(per_page)

    # Button cho tìm kiếm gõ tay
    if st.button("🔍 Tìm kiếm"):
        if str(q_input).strip() == "":
            st.info("Hãy nhập id hoặc từ khóa vào ô tìm kiếm.")
        else:
            st.session_state.page = 1
            st.session_state.last_query = str(q_input).strip()
            st.session_state.last_query_method = "typed"

    # ------------------ Processing search ------------------
    last_q = st.session_state.get("last_query", "")
    method = st.session_state.get("last_query_method", "")
    if last_q:
        chosen_index = None
        chosen_method = None

        if method == "selectbox":
            # last_q là id
            try:
                q_num = int(last_q)
                matches = df_bikes.index[df_bikes["id"] == q_num].tolist()
                if matches:
                    chosen_index = matches[0]
                    chosen_method = f"id chính xác ({q_num})"
                else:
                    st.warning(f"Không tìm thấy id = {q_num} trong dữ liệu.")
            except Exception:
                st.warning("ID chọn không hợp lệ.")
        else:
            # typed: có thể là id hoặc từ khóa
            if last_q.isdigit():
                q_num = int(last_q)
                matches = df_bikes.index[df_bikes["id"] == q_num].tolist()
                if matches:
                    chosen_index = matches[0]
                    chosen_method = f"id chính xác ({q_num})"
            if chosen_index is None:
                best_idx, best_score = find_best_title_match(
                    df_bikes["Tiêu đề"].astype(str).tolist(), last_q
                )
                if best_idx is not None and best_score > 0.05:
                    chosen_index = best_idx
                    chosen_method = f"closest title match (score={best_score:.3f})"
                else:
                    st.warning(
                        "Không tìm thấy Tiêu đề nào giống query. Hãy thử từ khóa khác."
                    )
                    chosen_index = None

        # Nếu tìm được index -> dùng cosine để lấy gợi ý
        if chosen_index is not None:
            st.success(
                f"Đã chọn item index = {chosen_index} bằng phương pháp: {chosen_method}"
            )

            recommendations = get_recommendations_by_index(
                df_bikes,
                cosine_sim,
                chosen_index,
                top_k=st.session_state.max_results,
            )
            if recommendations.empty:
                st.write("_Không có gợi ý_")
            else:
                total = len(recommendations)
                total_pages = max(1, ceil(total / st.session_state.per_page))
                st.write(
                    f"Tổng gợi ý thu được: **{total}** — "
                    f"Hiển thị **{st.session_state.per_page}** / trang — "
                    f"Tổng trang: **{total_pages}**"
                )

                # normalize page in session_state
                if st.session_state.page < 1:
                    st.session_state.page = 1
                if st.session_state.page > total_pages:
                    st.session_state.page = total_pages

                # page chooser
                new_page = st.number_input(
                    "Chọn trang",
                    min_value=1,
                    max_value=total_pages,
                    value=st.session_state.page,
                    step=1,
                    key="ui_page",
                )
                if new_page != st.session_state.page:
                    st.session_state.page = int(new_page)

                # slice and display
                df_page = paginate_dataframe(
                    recommendations,
                    st.session_state.page,
                    st.session_state.per_page,
                )
                display_rows_with_expander(df_page)

                # navigation buttons
                nav_col1, nav_col2, _ = st.columns([1, 1, 4])
                with nav_col1:
                    if st.button("<< Trang trước"):
                        st.session_state.page = max(1, st.session_state.page - 1)
                with nav_col2:
                    if st.button("Trang sau >>"):
                        st.session_state.page = min(
                            total_pages, st.session_state.page + 1
                        )

    st.markdown("---")
    st.caption(
        "Ghi chú: Ma trận cosine được xây dựng từ nội dung Tiêu đề + Mô tả chi tiết. "
        "Đảm bảo thứ tự dòng giữa dataframe và ma trận là giống nhau (df.reset_index(drop=True))."
    )


# ======================================================
# MAIN APP
# ======================================================

def main():
    st.set_page_config(
        page_title="Project 2 – Đề xuất & phân khúc xe máy",
        layout="wide",
    )

    # Header chung
    render_header()

    # Sidebar: chọn chức năng
    st.sidebar.markdown("## Menu")
    page = st.sidebar.radio(
        "Chọn chức năng:",
        [
            "Giới thiệu dự án",
            "Đánh giá & báo cáo (KMeans)",
            "Khám phá & dự đoán phân khúc",
            "Tìm kiếm & đề xuất xe tương tự",
            "Thông tin nhóm",
        ],
    )

    # Với các trang liên quan phân cụm, cần load dữ liệu 1 lần
    cluster_data_path = get_file_path(CLUSTER_DATA_FILE)

    # Điều hướng
    if page == "Giới thiệu dự án":
        page_project_intro()

    elif page in ["Đánh giá & báo cáo (KMeans)", "Khám phá & dự đoán phân khúc"]:
        if not os.path.exists(cluster_data_path):
            st.error(
                f"❌ Không tìm thấy file dữ liệu phân cụm: {CLUSTER_DATA_FILE}. "
                "Hãy kiểm tra lại tên file."
            )
            return

        # Cho phép chọn K ở sidebar
        K = st.sidebar.slider(
            "Số phân khúc (K)", min_value=2, max_value=8, value=3
        )

        df, numeric_cols, categorical_cols, preprocess, X_dense = (
            load_and_prepare_cluster_data(cluster_data_path)
        )
        result = run_kmeans(df, numeric_cols, X_dense, K)

        if page == "Đánh giá & báo cáo (KMeans)":
            page_evaluation(result)
        else:
            page_cluster_and_predict(
                df, numeric_cols, categorical_cols, preprocess, result
            )

    elif page == "Tìm kiếm & đề xuất xe tương tự":
        page_search_and_recommend()

    else:
        page_team()


if __name__ == "__main__":
    main()
