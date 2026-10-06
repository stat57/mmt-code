# AFHQv2 64x64: EDM sampling with the denoised prediction recorded at every sampler step,
# CLIP and Inception features, PCA fitted on the calibration split, real and generated images.
# Kaggle: GPU T4 x2, Internet on.
# Output: /kaggle/working/results_core.zip and /kaggle/working/results_images.zip

import os, sys, re, glob, time, zipfile, subprocess, traceback
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, torch.nn.functional as F

N_SAMPLES = 5000
SEED0 = 20270001
SPLIT_SEED = 20261001
MODELS = ["vp", "ve"]
N_STEPS = 40
ENCODERS = ["clip", "inception"]
N_PCA = 50
N_FULL = 1000
N_TRAJ_IMG = 200
PREFLIGHT_N = 8
GEN_BATCH_PER_GPU = 125
EMB_BATCH_PER_GPU = 256
RES = 64
EDM_URL = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-afhqv2-64x64-uncond-{}.pkl"

WORK = "/kaggle/working"
TMP = "/tmp/mmt"
OUT = WORK + "/results"
IMG = TMP + "/imgs"
os.makedirs(TMP, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
os.makedirs(IMG, exist_ok=True)
T0 = time.time()
LOGF = open(OUT + "/run_log.txt", "a")


def log(msg):
    line = "[%7.1f min] %s" % ((time.time() - T0) / 60, msg)
    print(line, flush=True)
    LOGF.write(line + "\n")
    LOGF.flush()


def zip_results():
    groups = {"core": [], "images": []}
    for p in sorted(glob.glob(OUT + "/*")):
        groups["images" if os.path.basename(p).startswith("images_") else "core"].append(p)
    for name, files in groups.items():
        with zipfile.ZipFile(WORK + "/results_%s.zip" % name, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in files:
                zf.write(p, os.path.basename(p))


assert torch.cuda.is_available(), "GPU required"
NGPU = torch.cuda.device_count()
DEVS = ["cuda:%d" % i for i in range(NGPU)]
torch.backends.cudnn.benchmark = True
log("GPUs: %d x %s" % (NGPU, torch.cuda.get_device_name(0)))


class MultiGPU:
    def __init__(self, factory, amp):
        self.models = [factory(d) for d in DEVS]
        self.amp = amp
        self.pool = ThreadPoolExecutor(max_workers=NGPU)

    def _run(self, i, fn, chunks):
        dev = DEVS[i]
        with torch.cuda.device(dev), torch.no_grad(), \
                torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            args = [c.to(dev, non_blocking=True) for c in chunks]
            return fn(self.models[i], *args)

    def __call__(self, fn, *tensors):
        k = min(NGPU, tensors[0].shape[0])
        if k == 1:
            return self._run(0, fn, tensors).to(DEVS[0])
        parts = [torch.tensor_split(t, k) for t in tensors]
        futs = [self.pool.submit(self._run, i, fn, [p[i] for p in parts]) for i in range(k)]
        return torch.cat([f.result().to(DEVS[0]) for f in futs])

    def free(self):
        self.pool.shutdown()
        del self.models
        torch.cuda.empty_cache()


# ---------------- data ----------------

def load_afhq():
    import PIL.Image
    root = TMP + "/afhq"
    if not glob.glob(root + "/**/train/cat/*", recursive=True):
        if not os.path.exists(TMP + "/stargan-v2"):
            subprocess.run(["git", "clone", "-q", "https://github.com/clovaai/stargan-v2",
                            TMP + "/stargan-v2"], check=True)
        sh = open(TMP + "/stargan-v2/download.sh").read()
        url = re.search(r"URL=(\S+)", sh[sh.index("afhq-v2-dataset"):]).group(1).replace("dl=0", "dl=1")
        import urllib.request
        urllib.request.urlretrieve(url, TMP + "/afhq_v2.zip")
        with zipfile.ZipFile(TMP + "/afhq_v2.zip") as z:
            z.extractall(root)
        os.remove(TMP + "/afhq_v2.zip")
    data = {}
    for split in ["train", "test"]:
        files = sorted(glob.glob(root + "/**/" + split + "/*/*.png", recursive=True)
                       + glob.glob(root + "/**/" + split + "/*/*.jpg", recursive=True))
        labels = np.array([["cat", "dog", "wild"].index(os.path.basename(os.path.dirname(p))) for p in files])
        ims = np.stack([np.asarray(PIL.Image.open(p).convert("RGB").resize((RES, RES), PIL.Image.LANCZOS))
                        for p in files]).astype(np.uint8)
        ids = np.array([os.path.join(os.path.basename(os.path.dirname(p)), os.path.basename(p)) for p in files])
        data[split] = (ims, labels, ids)
    return data


# ---------------- encoders ----------------

def load_encoder(name):
    if name == "clip":
        from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
        proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

        def fac(dev):
            return CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()

        def prep(arr):
            return proc(images=list(arr), return_tensors="pt")["pixel_values"]

        def fn(m, x):
            return m(pixel_values=x).image_embeds.float()

        return MultiGPU(fac, amp=True), prep, fn
    if name == "inception":
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytorch-fid==0.3.0"], check=True)
        from pytorch_fid.inception import InceptionV3

        def fac(dev):
            return InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()

        def prep(arr):
            return torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float() / 255.0

        def fn(m, x):
            return m(x)[0].flatten(1).float()

        return MultiGPU(fac, amp=False), prep, fn
    raise ValueError(name)


def embed(enc, prep, fn, arr):
    out, step = [], EMB_BATCH_PER_GPU * NGPU
    for i in range(0, len(arr), step):
        x = prep(np.asarray(arr[i:i + step])).to(DEVS[0])
        out.append(enc(fn, x).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ---------------- EDM sampling ----------------

def edm_code():
    if not os.path.exists(TMP + "/edm"):
        subprocess.run(["git", "clone", "-q", "https://github.com/NVlabs/edm", TMP + "/edm"], check=True)
    if TMP + "/edm" not in sys.path:
        sys.path.insert(0, TMP + "/edm")


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def initial_noise(seeds):
    z = [torch.randn(3, RES, RES, generator=torch.Generator().manual_seed(int(s))) for s in seeds]
    return torch.stack(z).to(DEVS[0])


def run_edm(kind, n, path=None):
    edm_code()
    import pickle, dnnlib

    def fac(dev):
        with dnnlib.util.open_url(EDM_URL.format(kind)) as f:
            return pickle.load(f)["ema"].to(dev).eval()

    net = MultiGPU(fac, amp=False)
    assert net.models[0].img_resolution == RES

    def den_fn(m, x, s):
        return m(x, s).float()

    smin, smax, rho = 0.002, 80.0, 7.0
    i = torch.arange(N_STEPS, dtype=torch.float64, device=DEVS[0])
    ts = (smax ** (1 / rho) + i / (N_STEPS - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    ts = torch.cat([net.models[0].round_sigma(ts), torch.zeros_like(ts[:1])])
    steps_sigma = ts[:-1].cpu().numpy()
    shape = (N_STEPS + 1, n, RES, RES, 3)
    if path is None:
        imgs = np.empty(shape, np.uint8)
    else:
        imgs = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    B = GEN_BATCH_PER_GPU * NGPU
    for b in range(0, n, B):
        nb = min(B, n - b)
        x_next = initial_noise(range(SEED0 + b, SEED0 + b + nb)).double() * ts[0]
        for k in range(N_STEPS):
            tc, tn = ts[k], ts[k + 1]
            x_cur = x_next
            den = net(den_fn, x_cur.float(), tc.float().expand(nb).contiguous()).double()
            imgs[k, b:b + nb] = to_uint8(den)
            d_cur = (x_cur - den) / tc
            x_next = x_cur + (tn - tc) * d_cur
            if k < N_STEPS - 1:
                den2 = net(den_fn, x_next.float(), tn.float().expand(nb).contiguous()).double()
                x_next = x_cur + (tn - tc) * (0.5 * d_cur + 0.5 * (x_next - den2) / tn)
        imgs[-1, b:b + nb] = to_uint8(x_next)
        log("%s: %d/%d" % (kind, b + nb, n))
    net.free()
    if path is not None:
        imgs.flush()
    return imgs, np.append(steps_sigma, 0.0)


def save_grid(name, imgs):
    from torchvision.utils import save_image
    cols = list(range(0, imgs.shape[0] - 1, 4)) + [imgs.shape[0] - 1]
    g = torch.from_numpy(np.asarray(imgs[cols, :8])).permute(1, 0, 4, 2, 3).reshape(-1, 3, RES, RES).float() / 255.0
    save_image(g, OUT + "/grid_" + name + ".png", nrow=len(cols), padding=1)


# ---------------- preflight ----------------

report, ok = [], True


def check(label, f):
    global ok
    t = time.time()
    try:
        msg = f()
        report.append("PASS  %-26s %s  (%.0fs)" % (label, msg, time.time() - t))
    except Exception:
        ok = False
        report.append("FAIL  %-26s\n%s" % (label, traceback.format_exc()))
    log(report[-1].splitlines()[0])


log("PREFLIGHT")
DATA = {}


def _data():
    DATA.update(load_afhq())
    tr, te = DATA["train"], DATA["test"]
    assert len(tr[0]) > 10000 and len(te[0]) > 1000
    return "train %s %s, test %s" % (tr[0].shape, np.bincount(tr[1]).tolist(), te[0].shape)


check("AFHQv2 data", _data)

PRE = {}
for m in MODELS:
    def _gen(m=m):
        imgs, sig = run_edm(m, PREFLIGHT_N)
        assert imgs.std() > 1
        PRE[m] = imgs
        return "sigmas %s" % np.round(sig, 3).tolist()
    check("model " + m, _gen)

for e in ENCODERS:
    def _enc(e=e):
        enc, prep, fn = load_encoder(e)
        f1 = embed(enc, prep, fn, DATA["test"][0][:32])
        f2 = embed(enc, prep, fn, PRE[MODELS[0]].reshape(-1, RES, RES, 3))
        enc.free()
        assert np.all(np.isfinite(f1)) and np.all(np.isfinite(f2)) and f1.std() > 0
        return "dim %d" % f1.shape[1]
    check("encoder " + e, _enc)

open(OUT + "/preflight_report.txt", "w").write("\n".join(report))
zip_results()
if not ok:
    log("PREFLIGHT FAILED")
    sys.exit(0)
log("PREFLIGHT PASSED")

# ---------------- splits ----------------

Xtr, ytr, idtr = DATA["train"]
Xte, yte, idte = DATA["test"]
perm = np.random.default_rng(SPLIT_SEED).permutation(len(Xtr))
half = len(perm) // 2
idx_A, idx_B = np.sort(perm[:half]), np.sort(perm[half:])
np.savez_compressed(OUT + "/splits.npz", idx_A=idx_A, idx_B=idx_B, y_A=ytr[idx_A], y_B=ytr[idx_B], y_test=yte,
                    id_A=idtr[idx_A], id_B=idtr[idx_B], id_test=idte, split_seed=SPLIT_SEED)
np.savez_compressed(OUT + "/images_real.npz", train=Xtr, test=Xte, y_train=ytr, y_test=yte,
                    id_train=idtr, id_test=idte)
zip_results()

# ---------------- generation ----------------

SIG = {}
for m in MODELS:
    try:
        os.makedirs(IMG, exist_ok=True)
        imgs, sig = run_edm(m, N_SAMPLES, IMG + "/" + m + ".npy")
        SIG[m] = sig
        save_grid(m, imgs)
        np.savez_compressed(OUT + "/images_gen_%s.npz" % m, final=np.asarray(imgs[-1]),
                            trajectories=np.asarray(imgs[:, :N_TRAJ_IMG]), sigma=sig.astype(np.float32),
                            seeds=np.arange(SEED0, SEED0 + N_SAMPLES), trajectory_seeds=np.arange(SEED0, SEED0 + N_TRAJ_IMG))
        log("%s done, sigmas %s" % (m, np.round(sig, 3).tolist()))
        del imgs
    except Exception:
        open(OUT + "/ERROR_generate_" + m + ".txt", "w").write(traceback.format_exc())
        log("%s generation failed" % m)
    zip_results()

# ---------------- embedding ----------------

from sklearn.decomposition import PCA

for e in ENCODERS:
    try:
        enc, prep, fn = load_encoder(e)
        FA = embed(enc, prep, fn, Xtr[idx_A])
        FB = embed(enc, prep, fn, Xtr[idx_B])
        FT = embed(enc, prep, fn, Xte)
        pca = PCA(n_components=N_PCA, svd_solver="full").fit(FA)
        np.savez_compressed(
            OUT + "/real_" + e + ".npz",
            S_A=pca.transform(FA).astype(np.float32), S_B=pca.transform(FB).astype(np.float32),
            S_test=pca.transform(FT).astype(np.float32),
            F_A=FA.astype(np.float16), F_B=FB.astype(np.float16), F_test=FT.astype(np.float16),
            pca_mean=pca.mean_.astype(np.float32), pca_components=pca.components_.astype(np.float32),
            pca_explained_variance=pca.explained_variance_.astype(np.float32))
        log("%s real features done" % e)
        for m in SIG:
            imgs = np.load(IMG + "/" + m + ".npy", mmap_mode="r")
            L = imgs.shape[0]
            S = np.empty((L, N_SAMPLES, N_PCA), np.float32)
            Ffull = np.empty((L, N_FULL, FA.shape[1]), np.float16)
            for j in range(L):
                Fj = embed(enc, prep, fn, imgs[j])
                S[j] = pca.transform(Fj)
                Ffull[j] = Fj[:N_FULL]
                if j == L - 1:
                    Ffinal = Fj.astype(np.float16)
                if j % 10 == 0:
                    log("  %s %s stage %d/%d" % (e, m, j + 1, L))
            np.savez_compressed(OUT + "/gen_" + m + "_" + e + ".npz", S=S, F_first1000=Ffull, F_final=Ffinal,
                                sigma=SIG[m].astype(np.float32), seeds=np.arange(SEED0, SEED0 + N_SAMPLES))
            log("%s %s features done" % (e, m))
        enc.free()
    except Exception:
        open(OUT + "/ERROR_embed_" + e + ".txt", "w").write(traceback.format_exc())
        log("%s embedding failed" % e)
    zip_results()

log("ALL DONE")
zip_results()