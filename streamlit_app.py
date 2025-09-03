# streamlit_app.py
import os, io, requests
from PIL import Image
import streamlit as st
from huggingface_hub import hf_hub_download

# importa do seu módulo
from model import load_model, infer_image_pil

st.set_page_config(page_title="Contagem de Neurofibromas", layout="wide")
st.title("Contagem de Neurofibromas na Pele")

# ======================== Sidebar (config) ========================
with st.sidebar:
    st.header("Configurações do Modelo")
    repo_id = st.text_input("HF repo_id", value="jubarrossousa/visao_comp_model_mestrado")
    ckpt_file = st.text_input("HF filename", value="best_state_dict_only.pth")
    downsample = st.slider("downsample (mesmo do treino)", 1, 8, value=2, step=1)
    make_overlay = st.checkbox("Gerar imagem anotada (heatmap)", value=True)
    max_side = st.number_input(
        "Pré-redimensionar imagens (px, maior lado)", min_value=512, max_value=4096, value=1280, step=64,
        help="Acelera a inferência em CPU reduzindo a resolução de entrada."
    )
    
@st.cache_resource(show_spinner=True)
def load_model_cached(repo_id: str, ckpt_file: str):
    """
    Baixa o checkpoint do HF (usa HF_TOKEN se existir em st.secrets)
    e carrega o modelo apenas uma vez (cache).
    """
    token = st.secrets.get("HF_TOKEN", None)
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_file, token=token)
    model, device = load_model(ckpt_path, device="cpu")  # Streamlit Cloud roda em CPU
    return model, device, ckpt_path

# ======================== Entradas ========================
st.markdown("Envie **uma ou mais imagens**.")
uploads = st.file_uploader(
    "Upload de imagens", type=["png","jpg","jpeg","bmp","webp","tiff"], accept_multiple_files=True
)

# ======================== Botão principal ========================
if st.button("Rodar Predição", type="primary"):
    # Coleta imagens
    images, names = [], []
    if uploads:
        for f in uploads:
            try:
                img = Image.open(f).convert("RGB")
                images.append(img)
                names.append(os.path.basename(getattr(f, "name", "upload.jpg")))
            except Exception as e:
                st.warning(f"Falha ao abrir {getattr(f,'name','(sem nome)')}: {e}")

    if not images:
        st.warning("Envie pelo menos uma imagem.")
        st.stop()

    with st.spinner("Baixando checkpoint e carregando o modelo..."):
        try:
            model, device, ckpt_path = load_model_cached(repo_id, ckpt_file)
        except Exception as e:
            st.error(f"Não consegui carregar o checkpoint do Hugging Face: {e}")
            st.stop()

    # ======================== Inferência ========================
    with st.spinner("Rodando inferência..."):
        rows = []
        annotated = []
        total = 0.0

        for name, img in zip(names, images):
            # Pré-resize opcional para acelerar CPU
            if max_side:
                w, h = img.size
                scale = min(1.0, float(max_side) / max(w, h))
                if scale < 1.0:
                    img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.BILINEAR)

            try:
                count, overlay_img = infer_image_pil(
                    image=img,
                    model=model,
                    device=device,
                    downsample=int(downsample),
                    return_overlay=make_overlay
                )
                count = max(0.0, float(count))  # clamp para não-negativo
                rows.append({"arquivo": name, "contagem": round(count, 1)})
                total += count
                if make_overlay and overlay_img is not None:
                    annotated.append((name, overlay_img))
            except Exception as e:
                rows.append({"arquivo": name, "contagem": "erro"})
                st.warning(f"Falha ao inferir {name}: {e}")

    # ======================== Saídas ========================
    st.subheader("Resultados")
    st.caption(f"Checkpoint: `{ckpt_path}` • downsample={downsample} • device=cpu")
    st.dataframe(rows, hide_index=True, use_container_width=True)

    st.metric("Total (todas as imagens)", f"{total:.1f}")

    if annotated:
        st.subheader("Imagens anotadas")
        cols = st.columns(3)
        for i, (name, im) in enumerate(annotated):
            with cols[i % 3]:
                st.image(im, caption=name, use_container_width=True)
