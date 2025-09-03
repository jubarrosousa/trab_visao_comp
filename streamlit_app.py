# streamlit_app.py
import os
import io
import requests
from PIL import Image
import streamlit as st
from huggingface_hub import hf_hub_download

# seu módulo com o modelo/inferência
from model import load_model, infer_image_pil

st.set_page_config(page_title="Contagem de neurofibromas", layout="wide")
st.title("Contagem de neurofibromas")

# ======================== Sidebar (config) ========================
with st.sidebar:
    st.header("Configurações do Modelo")
    repo_id = st.text_input("HF repo_id", value="jubarrossousa/visao_comp_model_mestrado")
    ckpt_file = st.text_input("HF filename", value="best_state_dict_only.pth")
    downsample = st.slider("downsample (mesmo do treino)", 1, 8, value=2, step=1)
    make_overlay = st.checkbox("Gerar imagem anotada (heatmap)", value=True)
    st.caption("Se o repo do HF for privado, adicione **HF_TOKEN** em Settings → Secrets.")

@st.cache_resource(show_spinner=True)
def load_model_cached(repo_id: str, ckpt_file: str):
    """
    Baixa o checkpoint do HF (usa HF_TOKEN se existir em st.secrets)
    e carrega o modelo uma única vez (cache por instância).
    """
    token = st.secrets.get("HF_TOKEN", None)  # para repositórios privados use Secrets
    ckpt_path = hf_hub_download(repo_id=repo_id, filename=ckpt_file, token=token)
    model, device = load_model(ckpt_path, device="cpu")  # Streamlit Cloud roda em CPU
    return model, device, ckpt_path

def fetch_url_image(url: str) -> Image.Image:
    r = requests.get(url, timeout=12)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")

# ======================== Entradas ========================
st.markdown("Envie **uma ou mais imagens** ou cole **URLs** (uma por linha).")
uploads = st.file_uploader(
    "Upload de imagens", type=["png", "jpg", "jpeg", "bmp", "webp", "tiff"],
    accept_multiple_files=True
)
urls_text = st.text_area(
    "URLs (opcional)", height=80,
    placeholder="https://exemplo.com/img1.jpg\nhttps://exemplo.com/img2.png"
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
    if urls_text.strip():
        for line in urls_text.splitlines():
            u = line.strip()
            if not u:
                continue
            try:
                img = fetch_url_image(u)
                images.append(img)
                base = os.path.basename(u.split("?")[0]) or f"url_{len(images)}.jpg"
                names.append(base)
            except Exception as e:
                st.warning(f"Falha ao baixar {u}: {e}")

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
        imgs = [im for _, im in annotated]
        caps = [nm for nm, _ in annotated]
        st.image(imgs, caption=caps, use_container_width=True)
