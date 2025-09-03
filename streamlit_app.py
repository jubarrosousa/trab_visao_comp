import streamlit as st
from PIL import Image
import os, io, uuid, shutil, csv, json, glob, requests, tempfile
from huggingface_hub import hf_hub_download

# ====== CONFIGURE AQUI ======
CKPT_REPO = "jubarrossousa/visao_comp_model_mestrado"     # repo no HF Hub
CKPT_FILE = "best_state_dict_only.pth"        # nome do arquivo de pesos
DEFAULT_DOWNSAMPLE = 2
WRITE_VIS_DEFAULT = True

# ====== IMPORTE SUA INFERÊNCIA DE PASTA ======
# Coloque sua função infer_folder(...) dentro de your_model.py (mesmo repo)
from your_model import infer_folder  # <- você fornece

st.set_page_config(page_title="Contagem de Bolinhas", layout="centered")
st.title("Contagem de Bolinhas na Pele")

@st.cache_resource(show_spinner=False)
def get_ckpt_path(repo_id: str, filename: str) -> str:
    """Baixa (uma vez) o checkpoint do Hugging Face Hub.
       Se o repo for privado, defina o secret HF_TOKEN nas settings do Streamlit."""
    token = st.secrets.get("HF_TOKEN", None)
    path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    return path

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _save_inputs(files, urls_text, input_dir):
    """Salva uploads/URLs no input_dir e retorna lista de caminhos na ordem."""
    _ensure_dir(input_dir)
    saved = []

    # Uploads
    if files:
        for f in files:
            try:
                img = Image.open(f).convert("RGB")
                name = os.path.basename(getattr(f, "name", f"upload_{uuid.uuid4().hex}.jpg"))
                if os.path.splitext(name)[1].lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"]:
                    name += ".jpg"
                out_path = os.path.join(input_dir, name)
                img.save(out_path)
                saved.append(out_path)
            except Exception as e:
                st.warning(f"Falha ao ler upload {getattr(f,'name','sem-nome')}: {e}")

    # URLs (uma por linha)
    if urls_text:
        for line in urls_text.splitlines():
            u = line.strip()
            if not u:
                continue
            try:
                r = requests.get(u, timeout=12)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                ext = os.path.splitext(u)[1].lower()
                if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"]:
                    ext = ".jpg"
                out_path = os.path.join(input_dir, f"url_{uuid.uuid4().hex}{ext}")
                img.save(out_path)
                saved.append(out_path)
            except Exception as e:
                st.warning(f"Falha ao baixar URL {u}: {e}")
    return saved

def _try_load_counts(out_dir):
    """Procura por counts.csv/preds.csv/results.csv (filename,count) ou counts.json, ou *.txt."""
    # CSVs
    for name in ["counts.csv", "preds.csv", "results.csv"]:
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            out = {}
            with open(p, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        fn, val = row[0], row[1]
                        try:
                            out[os.path.basename(fn)] = max(0, int(round(float(val))))
                        except:
                            pass
            if out:
                return out

    # JSON
    p = os.path.join(out_dir, "counts.json")
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            out = {}
            for k, v in d.items():
                try:
                    out[os.path.basename(k)] = max(0, int(round(float(v))))
                except:
                    pass
            if out:
                return out
        except:
            pass

    # TXT por imagem (ex: img123.txt com "17")
    out = {}
    for p in glob.glob(os.path.join(out_dir, "*.txt")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                val = f.read().strip().split()[0]
                c = max(0, int(round(float(val))))
                key = os.path.basename(os.path.splitext(p)[0])
                out[key] = c
        except:
            pass
    return out

def _gather_outputs(input_paths, out_dir):
    """Retorna (rows_df, images_annotated, total) na mesma ordem dos inputs."""
    counts_map = _try_load_counts(out_dir)
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tiff"]

    rows = []
    annotated_imgs = []
    total = 0
    have_counts = False

    for p in input_paths:
        base = os.path.basename(p)
        base_noext = os.path.splitext(base)[0]

        # contagem
        c = None
        if base in counts_map:
            c = counts_map[base]
            have_counts = True
        elif base_noext in counts_map:
            c = counts_map[base_noext]
            have_counts = True
        rows.append({"arquivo": base, "contagem": (c if c is not None else "N/A")})
        if isinstance(c, int):
            total += c

        # imagem anotada
        ann = None
        for e in exts:
            cand = os.path.join(out_dir, base_noext + e)
            if os.path.exists(cand):
                try:
                    ann = Image.open(cand).convert("RGB")
                    break
                except:
                    pass
        if ann is None:
            # fallback: qualquer imagem que comece com o basename
            for cnd in glob.glob(os.path.join(out_dir, base_noext + "*")):
                if os.path.splitext(cnd)[1].lower() in exts:
                    try:
                        ann = Image.open(cnd).convert("RGB")
                        break
                    except:
                        pass
        if ann is not None:
            annotated_imgs.append(ann)

    return rows, annotated_imgs, (total if have_counts else None)

with st.sidebar:
    st.header("Configurações")
    repo_id = st.text_input("HF repo_id", CKPT_REPO)
    ckpt_file = st.text_input("HF filename", CKPT_FILE)
    device = st.selectbox("Device", ["cpu"], index=0)  # Streamlit Cloud não tem GPU
    downsample = st.slider("downsample", 1, 8, value=DEFAULT_DOWNSAMPLE, step=1)
    write_vis = st.checkbox("Gerar imagens anotadas (write_vis)", value=WRITE_VIS_DEFAULT)

st.markdown("Envie **uma ou mais imagens** ou cole **URLs** (uma por linha).")
uploads = st.file_uploader("Upload de imagens", type=["png","jpg","jpeg","bmp","webp","tiff"], accept_multiple_files=True)
urls_text = st.text_area("URLs (opcional)", placeholder="https://exemplo.com/img1.jpg\nhttps://exemplo.com/img2.png", height=80)

if st.button("Rodar Predição", type="primary"):
    with st.spinner("Baixando checkpoint e rodando inferência..."):
        try:
            ckpt_path = get_ckpt_path(repo_id, ckpt_file)
        except Exception as e:
            st.error(f"Não consegui baixar o checkpoint do HF: {e}")
            st.stop()

        # diretórios temporários por request
        base_dir = tempfile.mkdtemp(prefix="infer_")
        in_dir = os.path.join(base_dir, "in"); os.makedirs(in_dir, exist_ok=True)
        out_dir = os.path.join(base_dir, "out"); os.makedirs(out_dir, exist_ok=True)

        input_paths = _save_inputs(uploads, urls_text, in_dir)
        if not input_paths:
            st.warning("Nenhuma imagem válida enviada.")
            shutil.rmtree(base_dir, ignore_errors=True)
            st.stop()

        try:
            # sua função (você fornece em your_model.py)
            infer_folder(
                ckpt_path=ckpt_path,
                images_dir=in_dir,
                out_dir=out_dir,
                downsample=int(downsample),
                device=device,
                write_vis=bool(write_vis),
            )
        except Exception as e:
            st.error(f"Falha ao executar infer_folder: {e}")
            shutil.rmtree(base_dir, ignore_errors=True)
            st.stop()

        rows, ann_imgs, total = _gather_outputs(input_paths, out_dir)
        shutil.rmtree(base_dir, ignore_errors=True)

    # Mostrar resultados
    if rows:
        st.subheader("Contagem por imagem")
        st.dataframe(rows, use_container_width=True)
    if total is not None:
        st.metric("Total (todas as imagens)", total)
    if ann_imgs:
        st.subheader("Imagens anotadas")
        st.image(ann_imgs, caption=[os.path.basename(p) for p in input_paths], use_column_width=True)
