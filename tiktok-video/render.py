import json, math, subprocess, wave, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
NAVY, NAVY2 = (12, 18, 38), (22, 32, 64)
GOLD, GOLD2 = (214, 172, 84), (245, 214, 140)
WHITE, SOFT = (250, 248, 242), (190, 196, 214)
F = lambda n, s: ImageFont.truetype(f"Poppins-{n}.ttf", s)

scenes = json.load(open("scenes.json"))
GAP = 0.25
t0 = 0.0
for i, s in enumerate(scenes):
    s["start"] = t0
    s["len"] = min(s["dur"], s["words"][-1][1] + 0.45)
    t0 += s["len"] + GAP
TOTAL = t0 + 0.8

# ---------- audio: concat scenes, trimmed, + loudness envelope for lip-sync
SR = 44100
voice = np.zeros(int(TOTAL * SR), np.float32)
for i, s in enumerate(scenes):
    with wave.open(f"s{i}.wav") as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    a = a[: int(s["len"] * SR)]
    st = int(s["start"] * SR)
    voice[st:st + len(a)] += a
spf = SR // FPS
env = np.array([np.sqrt(np.mean(voice[k*spf:(k+1)*spf] ** 2) + 1e-9) for k in range(int(TOTAL*FPS)+2)])
env = np.nan_to_num(env)
env = np.clip(env / (np.percentile(env[env > 0.01], 90) + 1e-6), 0, 1.2)
env = np.convolve(env, [0.25, 0.5, 0.25], "same")

# soft ambient pad (A minor add9), very low volume
t = np.arange(len(voice)) / SR
pad = sum(np.sin(2*np.pi*f*t) * a for f, a in [(110, .5), (164.8, .35), (220, .3), (246.9, .18), (329.6, .15)])
pad *= (0.6 + 0.4*np.sin(2*np.pi*0.12*t)) * 0.035
fade = np.minimum(1, np.minimum(t / 1.5, (TOTAL - t) / 1.5))
mix = voice * 1.0 + pad * fade
mix = np.clip(mix, -1, 1)
with wave.open("mix.wav", "w") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((mix * 32767).astype(np.int16).tobytes())

# ---------- helpers
def ease(x): x = max(0, min(1, x)); return 1 - (1 - x) ** 3
def lerp(a, b, x): return a + (b - a) * x

def text_c(d, y, txt, font, fill, cx=W//2, stroke=0, sfill=None):
    w = d.textlength(txt, font=font)
    d.text((cx - w/2, y), txt, font=font, fill=fill, stroke_width=stroke, stroke_fill=sfill)

def rounded(im, r):
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, *im.size), r, fill=255)
    im = im.convert("RGBA"); im.putalpha(m); return im

def shadowed(im, blur=30, off=18, alpha=150):
    pad = blur * 2
    sh = Image.new("RGBA", (im.width + pad*2, im.height + pad*2), (0, 0, 0, 0))
    a = im.split()[-1].point(lambda v: v * alpha // 255)
    blk = Image.new("RGBA", im.size, (0, 0, 0, 255)); blk.putalpha(a)
    sh.paste(blk, (pad, pad + off), blk)
    sh = sh.filter(ImageFilter.GaussianBlur(blur))
    sh.alpha_composite(im, (pad, pad))
    return sh, pad

def paste_c(base, im, cx, cy, alpha=1.0):
    if alpha < 1:
        im = im.copy(); im.putalpha(im.split()[-1].point(lambda v: int(v * alpha)))
    base.alpha_composite(im, (int(cx - im.width/2), int(cy - im.height/2)))

def scaled(im, s):
    return im.resize((max(1, int(im.width*s)), max(1, int(im.height*s))), Image.LANCZOS)

# ---------- background
bg = Image.new("RGBA", (W, H))
g = np.linspace(0, 1, H)[:, None]
arr = np.zeros((H, W, 3))
for c in range(3): arr[..., c] = lerp(NAVY2[c], NAVY[c], g)
yy, xx = np.mgrid[0:H, 0:W]
glow = np.exp(-(((xx - 540)/700)**2 + ((yy - 520)/600)**2))
arr += glow[..., None] * np.array([40, 30, 10])
bg = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).convert("RGBA")
rng = np.random.default_rng(3)
particles = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(1.5, 4), rng.uniform(8, 30), rng.uniform(0, 6.28)) for _ in range(45)]

# ---------- assets
page0 = Image.open("img0.png").convert("RGB")
page1 = Image.open("img1.png").convert("RGB")
photo = Image.open("img3.png").convert("RGB")
cover_card, cover_pad = shadowed(rounded(page0.resize((560, 792), Image.LANCZOS), 18), 34, 22, 170)
cover_small, _ = shadowed(rounded(page0.resize((380, 537), Image.LANCZOS), 14), 24, 14, 170)
page1_card, _ = shadowed(rounded(page1.resize((500, 707), Image.LANCZOS), 16), 30, 18, 170)
ph = photo.resize((820, 820), Image.LANCZOS)
photo_card, _ = shadowed(rounded(ph, 36), 34, 20, 170)

# ---------- avatar (drawn at 2x then downsampled for anti-aliasing)
AV = 520
SKIN, SKIN_SH = (236, 196, 164), (214, 168, 136)
HAIR = (44, 30, 24)
SUIT, SUIT2 = (26, 30, 44), (38, 44, 64)

def avatar(t, level, frame):
    S = 2; A = AV * S
    im = Image.new("RGBA", (A, A), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    bob = math.sin(t * 2.1) * 6 * S + level * 5 * S
    tilt = math.sin(t * 0.9) * 8 * S
    cx, cy = A/2 + tilt*0.3, A*0.42 + bob
    # body / suit
    d.rounded_rectangle((A*0.14, A*0.74 + bob*0.4, A*0.86, A*1.25), 150*S, fill=SUIT)
    d.polygon([(A*0.42, A*0.74+bob*0.4), (A*0.58, A*0.74+bob*0.4), (A*0.5, A*0.92+bob*0.4)], fill=WHITE)
    d.polygon([(A*0.47, A*0.78+bob*0.4), (A*0.53, A*0.78+bob*0.4), (A*0.515, A*0.93+bob*0.4), (A*0.485, A*0.93+bob*0.4)], fill=GOLD)
    d.polygon([(A*0.30, A*0.76+bob*0.4), (A*0.42, A*0.74+bob*0.4), (A*0.5, A*0.95+bob*0.4), (A*0.40, A*1.0)], fill=SUIT2)
    d.polygon([(A*0.70, A*0.76+bob*0.4), (A*0.58, A*0.74+bob*0.4), (A*0.5, A*0.95+bob*0.4), (A*0.60, A*1.0)], fill=SUIT2)
    # neck
    d.rounded_rectangle((cx - 42*S, cy + 90*S, cx + 42*S, cy + 175*S), 20*S, fill=SKIN_SH)
    # ears
    for sx in (-1, 1):
        d.ellipse((cx + sx*118*S - 20*S, cy - 5*S, cx + sx*118*S + 20*S, cy + 50*S), fill=SKIN_SH)
    # head
    d.ellipse((cx - 120*S, cy - 140*S, cx + 120*S, cy + 150*S), fill=SKIN)
    # hair
    d.chord((cx - 128*S, cy - 160*S, cx + 128*S, cy + 60*S), 180, 360, fill=HAIR)
    d.ellipse((cx - 110*S, cy - 165*S, cx + 40*S, cy - 60*S), fill=HAIR)
    d.ellipse((cx - 30*S, cy - 170*S, cx + 125*S, cy - 70*S), fill=HAIR)
    d.rectangle((cx - 128*S, cy - 50*S, cx - 108*S, cy + 10*S), fill=HAIR)
    d.rectangle((cx + 108*S, cy - 50*S, cx + 128*S, cy + 10*S), fill=HAIR)
    # eyes + blink
    blink = (frame % 110) < 4
    brow = level * 6 * S
    for sx in (-1, 1):
        ex, ey = cx + sx*48*S, cy + 10*S
        d.line((ex - 26*S, ey - 38*S - brow, ex + 26*S, ey - 42*S - brow + (sx*4*S)), fill=HAIR, width=9*S)
        if blink:
            d.line((ex - 18*S, ey, ex + 18*S, ey), fill=HAIR, width=6*S)
        else:
            d.ellipse((ex - 20*S, ey - 15*S, ex + 20*S, ey + 15*S), fill=WHITE)
            d.ellipse((ex - 11*S, ey - 12*S, ex + 11*S, ey + 12*S), fill=(58, 40, 30))
            d.ellipse((ex - 4*S, ey - 8*S, ex + 3*S, ey - 1*S), fill=WHITE)
    # nose
    d.line((cx, cy + 20*S, cx - 8*S, cy + 58*S, cx + 6*S, cy + 60*S), fill=SKIN_SH, width=6*S, joint="curve")
    # mouth (lip-sync)
    o = max(0.0, min(1.0, level))
    mw, mh = (34 + 6*o)*S, (6 + 30*o)*S
    my = cy + 92*S
    if o < 0.08:
        d.arc((cx - 36*S, my - 22*S, cx + 36*S, my + 12*S), 20, 160, fill=(150, 70, 60), width=7*S)
    else:
        d.rounded_rectangle((cx - mw, my - mh/2, cx + mw, my + mh/2), int(mh/2), fill=(110, 30, 36))
        if mh > 16*S:
            d.rectangle((cx - mw*0.7, my - mh/2, cx + mw*0.7, my - mh/2 + 8*S), fill=WHITE)
    # glasses (expert look)
    for sx in (-1, 1):
        ex, ey = cx + sx*48*S, cy + 10*S
        d.rounded_rectangle((ex - 36*S, ey - 28*S, ex + 36*S, ey + 28*S), 16*S, outline=(30, 26, 26), width=6*S)
    d.line((cx - 12*S, cy + 4*S, cx + 12*S, cy + 4*S), fill=(30, 26, 26), width=6*S)
    im = im.resize((AV, AV), Image.LANCZOS)
    # circular frame
    frame_im = Image.new("RGBA", (AV, AV), (0, 0, 0, 0))
    fd = ImageDraw.Draw(frame_im)
    fd.ellipse((0, 0, AV-1, AV-1), fill=(30, 44, 86))
    frame_im.alpha_composite(im)
    m = Image.new("L", (AV, AV), 0); ImageDraw.Draw(m).ellipse((6, 6, AV-7, AV-7), fill=255)
    frame_im.putalpha(Image.eval(m, lambda v: v))
    out = Image.new("RGBA", (AV + 40, AV + 40), (0, 0, 0, 0))
    od = ImageDraw.Draw(out)
    ring = 4 + level * 10
    od.ellipse((20 - ring, 20 - ring, 20 + AV + ring, 20 + AV + ring), fill=(*GOLD, int(90 + 80*level)))
    od.ellipse((16, 16, AV + 24, AV + 24), fill=GOLD)
    out.alpha_composite(frame_im, (20, 20))
    return out

# ---------- captions: groups of ~3 words, active word in gold
cap_groups = []
for s in scenes:
    ws = s["words"]
    for k in range(0, len(ws), 3):
        grp = ws[k:k+3]
        cap_groups.append([(s["start"] + a, s["start"] + b, txt) for a, b, txt in grp])

CAPF = F("ExtraBold", 66)
def draw_caption(base, t):
    cur = None
    for gi, grp in enumerate(cap_groups):
        end = cap_groups[gi+1][0][0] if gi + 1 < len(cap_groups) else grp[-1][1] + 0.3
        end = min(end, grp[-1][1] + 0.6)
        if grp[0][0] - 0.05 <= t < end: cur = grp; break
    if not cur: return
    d = ImageDraw.Draw(base)
    words = [w[2].upper() for w in cur]
    sp = d.textlength(" ", font=CAPF)
    widths = [d.textlength(w, font=CAPF) for w in words]
    total = sum(widths) + sp * (len(words) - 1)
    scale_fix = 1.0
    x = W/2 - total/2
    y = 1555
    pop = ease((t - cur[0][0]) / 0.12)
    for (a, b, _), w, wd in zip(cur, words, widths):
        active = a <= t < max(b, a + 0.15)
        col = GOLD2 if active else WHITE
        d.text((x, y - (6 if active else 0) * pop), w, font=CAPF, fill=col, stroke_width=7, stroke_fill=(8, 10, 20))
        x += wd + sp

# ---------- scene content
FH = F("ExtraBold", 92); FH2 = F("ExtraBold", 74); FM = F("Bold", 48); FS = F("Medium", 38); FB = F("Bold", 42)

def scene_hook(base, lt, s):
    d = ImageDraw.Draw(base)
    a = ease(lt / 0.35)
    stop = F("ExtraBold", int(230 * (1.4 - 0.4*a)))
    w = d.textlength("STOP", font=stop)
    d.rounded_rectangle((W/2 - w/2 - 50, 240, W/2 + w/2 + 50, 240 + 290), 40, fill=(200, 48, 48))
    text_c(d, 240 + 145 - stop.size*0.62, "STOP", stop, WHITE)
    b = ease((lt - 1.0) / 0.4)
    if b > 0:
        y = lerp(760, 690, b)
        text_c(d, y, "Tu utilises ChatGPT", FH2, WHITE)
        text_c(d, y + 95, "comme Google ?", FH2, GOLD2)
    c = ease((lt - 4.0) / 0.4)
    if c > 0:
        text_c(d, lerp(960, 920, c), "Tu passes à côté de l'essentiel.", FS, SOFT)

def scene_cred(base, lt, s):
    d = ImageDraw.Draw(base)
    a = ease(lt / 0.5)
    n = int(min(10, lt / 0.9 * 10)) if lt < 0.9 else 10
    big = F("ExtraBold", 300)
    text_c(d, lerp(300, 230, a), f"{n}", big, GOLD2)
    text_c(d, 590, "ANS D'EXPÉRIENCE", FM, WHITE)
    text_c(d, 660, "dans l'univers des ebooks", FS, SOFT)
    b = ease((lt - 3.2) / 0.4)
    if b > 0:
        for i, lab in enumerate(["LUS", "ANALYSÉS", "CRÉÉS"]):
            x = 230 + i * 310
            yy = lerp(860, 800, b)
            bw = d.textlength(lab, font=FB) + 60
            d.rounded_rectangle((x - bw/2, yy, x + bw/2, yy + 84), 42, outline=GOLD, width=4)
            text_c(d, yy + 14, lab, FB, GOLD2, cx=x)
    c = ease((lt - 6.2) / 0.3)
    if c > 0:
        text_c(d, 960, "Celui-là sort du lot.", FM, WHITE)

def scene_cover(base, lt, s):
    d = ImageDraw.Draw(base)
    a = ease(lt / 0.6)
    sc = lerp(0.8, 1.0, a) + lt * 0.006
    img = scaled(cover_card, sc * 0.95)
    paste_c(base, img, W/2, lerp(760, 600, a) - 10, alpha=a)
    b = ease((lt - 3.0) / 0.4)
    if b > 0:
        for i, lab in enumerate(["CLAIR", "SANS CHARABIA", "100% DÉBUTANTS"]):
            bw = d.textlength(lab, font=FS) + 44
            x = [210, 540, 870][i]
            y = lerp(1080, 1060, b)
            d.rounded_rectangle((x - bw/2, y, x + bw/2, y + 66), 33, fill=GOLD)
            text_c(d, y + 10, lab, FS, NAVY, cx=x)
    text_c(d, 130, "MAÎTRISE CHATGPT EN PARTANT DE ZÉRO", F("Bold", 40), GOLD2)

ITEMS = [("Méthode pas à pas", "Un prompt qui marche du 1er coup"),
         ("Bibliothèque de prompts", "Prêts à copier-coller"),
         ("Spécial études", "Résumer un cours en 2 minutes"),
         ("Spécial travail", "Mails & comptes rendus express")]
def scene_list(base, lt, s):
    d = ImageDraw.Draw(base)
    text_c(d, 150, "CE QUE TU OBTIENS", FH2, WHITE)
    d.rounded_rectangle((W/2 - 90, 250, W/2 + 90, 258), 4, fill=GOLD)
    # reveal times from keyword positions in narration
    keys = ["méthode", "bibliothèque", "études", "travail"]
    times = []
    for k in keys:
        tt = next((a for a, b, w in s["words"] if k in w.lower()), 0)
        times.append(tt - 0.2)
    for i, (h, sub) in enumerate(ITEMS):
        a = ease((lt - times[i]) / 0.35)
        if a <= 0: continue
        y = 320 + i * 185
        x0 = lerp(W + 50, 90, a)
        d.rounded_rectangle((x0, y, x0 + 900, y + 160), 30, fill=(255, 255, 255, 18) if False else (30, 42, 80), outline=(70, 86, 130), width=2)
        d.ellipse((x0 + 30, y + 40, x0 + 110, y + 120), fill=GOLD)
        d.line((x0 + 50, y + 80, x0 + 66, y + 98, x0 + 94, y + 62), fill=NAVY, width=10, joint="curve")
        d.text((x0 + 140, y + 28), h, font=FB, fill=WHITE)
        d.text((x0 + 140, y + 88), sub, font=FS, fill=SOFT)

def scene_check(base, lt, s):
    d = ImageDraw.Draw(base)
    a = ease(lt / 0.5)
    img = scaled(page1_card, lerp(0.85, 1.0, a))
    paste_c(base, img.rotate(lerp(-8, -4, a), resample=Image.BICUBIC, expand=True), W/2 + 90, 640, alpha=a)
    b = ease((lt - 0.5) / 0.4)
    if b > 0:
        x = lerp(-300, 60, b)
        d.rounded_rectangle((x, 250, x + 420, 330), 40, fill=GOLD)
        d.text((x + 34, 262), "BONUS", font=FB, fill=NAVY)
        d.rounded_rectangle((x, 350, x + 520, 520), 26, fill=(200, 48, 48))
        d.text((x + 30, 370), "Checklist des", font=FB, fill=WHITE)
        d.text((x + 30, 430), "erreurs à éviter", font=FB, fill=WHITE)

def scene_result(base, lt, s):
    d = ImageDraw.Draw(base)
    a = ease(lt / 0.5)
    img = scaled(photo_card, lerp(0.9, 1.0, a) + lt*0.01)
    paste_c(base, img, W/2, 560, alpha=a)
    b = ease((lt - 1.2) / 0.35)
    if b > 0:
        y = lerp(1060, 1010, b)
        bw = 860
        d.rounded_rectangle((W/2 - bw/2, y - 250, W/2 + bw/2, y + 20), 36, fill=(12, 18, 38, 230))
        text_c(d, y - 230, "+ PLUSIEURS HEURES", FH2, GOLD2)
        text_c(d, y - 130, "gagnées chaque semaine", FM, WHITE)

def scene_price(base, lt, s):
    d = ImageDraw.Draw(base)
    text_c(d, 190, "OFFRE DE LANCEMENT", FM, GOLD2)
    a = ease(lt / 0.4)
    old = F("Bold", 120)
    if a > 0:
        w = d.textlength("60€", font=old)
        text_c(d, 300, "60€", old, (150, 156, 176))
        k = ease((lt - 2.0) / 0.4)
        if k > 0:
            d.line((W/2 - w/2 - 20, 390, lerp(W/2 - w/2 - 20, W/2 + w/2 + 20, k), 370), fill=(220, 60, 60), width=14)
    b = ease((lt - 2.4) / 0.35)
    if b > 0:
        big = F("ExtraBold", int(220 * lerp(1.3, 1.0, b)))
        text_c(d, 520 - (big.size - 220) * 0.5, "29,99€", big, GOLD2)
        tag = "-50%"
        d.rounded_rectangle((770, 470, 960, 560), 45, fill=(200, 48, 48))
        text_c(d, 478, tag, FM, WHITE, cx=865)
    c = ease((lt - 5.3) / 0.35)
    if c > 0:
        for i, lab in enumerate(["Paiement unique", "Accès à vie"]):
            x = [320, 760][i]
            bw = d.textlength(lab, font=FB) + 70
            y = lerp(930, 900, c)
            d.rounded_rectangle((x - bw/2, y, x + bw/2, y + 86), 43, outline=GOLD, width=4)
            text_c(d, y + 16, lab, FB, WHITE, cx=x)

def scene_cta(base, lt, s):
    d = ImageDraw.Draw(base)
    a = ease(lt / 0.5)
    paste_c(base, cover_small.rotate(math.sin(lt*2)*2, resample=Image.BICUBIC, expand=True), W/2, lerp(620, 560, a), alpha=a)
    b = ease((lt - 0.4) / 0.4)
    if b > 0:
        pulse = 1 + 0.04 * math.sin(lt * 7)
        bw, bh = 700 * pulse, 130 * pulse
        y = 920
        d.rounded_rectangle((W/2 - bw/2, y, W/2 + bw/2, y + bh), bh/2, fill=GOLD)
        text_c(d, y + bh/2 - 42, "LIEN DANS MA BIO", F("ExtraBold", 64), NAVY)
    text_c(d, 150, "COMMENCE AUJOURD'HUI", FH2 if False else F("ExtraBold", 64), WHITE)

SCENE_FN = dict(hook=scene_hook, cred=scene_cred, cover=scene_cover, list=scene_list,
                check=scene_check, result=scene_result, price=scene_price, cta=scene_cta)

# ---------- render loop
N = int(TOTAL * FPS)
only = int(sys.argv[1]) if len(sys.argv) > 1 else None
ff = None
if only is None:
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-i", "mix.wav",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
                           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                           "video_tiktok_ebook_chatgpt.mp4"], stdin=subprocess.PIPE)
frames = range(N) if only is None else [only]
for fi in frames:
    t = fi / FPS
    base = bg.copy()
    d = ImageDraw.Draw(base)
    for (px, py, r, sp, ph0) in particles:
        y = (py - t * sp) % H
        al = int(60 + 60 * math.sin(t * 1.5 + ph0))
        d.ellipse((px - r, y - r, px + r, y + r), fill=(*GOLD, al))
    # current scene
    cur = None
    for s in scenes:
        if s["start"] - GAP/2 <= t < s["start"] + s["len"] + GAP/2 or (s is scenes[-1] and t >= s["start"]):
            cur = s
    if cur:
        layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        SCENE_FN[cur["name"]](layer, t - cur["start"], cur)
        # fade out at scene end
        out = 1 - ease((t - (cur["start"] + cur["len"])) / (GAP/2 + 0.01)) if cur is not scenes[-1] else 1
        if out < 1:
            layer.putalpha(layer.split()[-1].point(lambda v: int(v * out)))
        base.alpha_composite(layer)
    # avatar
    lvl = float(env[min(fi, len(env)-1)])
    av = avatar(t, lvl, fi)
    paste_c(base, av, W/2, 1300)
    # name tag
    d = ImageDraw.Draw(base)
    tag = "Dryss · Expert ebooks"
    tw = d.textlength(tag, font=FS)
    d.rounded_rectangle((W/2 - tw/2 - 30, 1470, W/2 + tw/2 + 30, 1530), 30, fill=NAVY, outline=GOLD, width=3)
    text_c(d, 1477, tag, FS, WHITE)
    draw_caption(base, t)
    # progress bar
    d.rectangle((0, 0, int(W * t / TOTAL), 8), fill=GOLD)
    rgb = base.convert("RGB")
    if only is not None:
        rgb.save(f"preview_{fi}.png")
    else:
        ff.stdin.write(rgb.tobytes())
        if fi % 150 == 0: print(f"{fi}/{N}", flush=True)
if ff:
    ff.stdin.close(); ff.wait()
    print("done", TOTAL)
