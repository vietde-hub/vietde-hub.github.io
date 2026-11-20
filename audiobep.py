import os
import re
import json
import wave
import hashlib
import asyncio
import urllib.request

# ==============================================================================
# CẤU HÌNH TẠO ÂM THANH CHUYÊN NGÀNH BẾP & PHỤ BẾP (4 GIỌNG MIỄN PHÍ)
# Đọc dữ liệu động từ: bep.html
# Xuất file vào: audio/bep/<tên_giọng>/
# ==============================================================================
HTML_SOURCE_FILE = "bep.html"
BASE_AUDIO_DIR = os.path.join("audio", "bep")
MANIFEST_FILE = os.path.join(BASE_AUDIO_DIR, ".audio_manifest.json")
PIPER_MODELS_DIR = "piper_models"

PIPER_VOICES = {
    "piper_hochdeutsch": {
        "name": "Piper Thorsten — Chuẩn Đức (High)",
        "model": "de_DE-thorsten-high.onnx",
        "url_onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx",
        "url_json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/high/de_DE-thorsten-high.onnx.json",
    },
    "piper_rostock": {
        "name": "Piper Thorsten — Miền Bắc / Rostock (Medium)",
        "model": "de_DE-thorsten-medium.onnx",
        "url_onnx": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx",
        "url_json": "https://huggingface.co/rhasspy/piper-voices/resolve/main/de/de_DE/thorsten/medium/de_DE-thorsten-medium.onnx.json",
    },
}

EDGE_VOICES = {
    "edge_hochdeutsch": ("Edge Neural — Chuẩn Đức (Nữ Katja)", "de-DE-KatjaNeural"),
    "edge_rostock": ("Edge Neural — Miền Bắc (Nam Conrad)", "de-DE-ConradNeural"),
}


# ==============================================================================
# 1. TRÍCH XUẤT ĐỘNG TOÀN BỘ CÂU TIẾNG ĐỨC TỪ BEP.HTML
# ==============================================================================
def _extract_js_array_block(content, prop_or_const_name):
    m = re.search(rf"\b{re.escape(prop_or_const_name)}\s*[:=]\s*\[", content)
    if not m:
        raise ValueError(
            f"Không tìm thấy '{prop_or_const_name}' trong {HTML_SOURCE_FILE}"
        )

    arr_start = content.index("[", m.start())
    depth = 0
    in_str = None
    escape = False

    for i in range(arr_start, len(content)):
        ch = content[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"', "`"):
            in_str = ch
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return content[arr_start : i + 1]

    raise ValueError(f"Không tìm thấy dấu đóng ngoặc mảng ']' cho {prop_or_const_name}")


def _split_top_level_objects(array_text):
    objects = []
    depth = 0
    obj_start = None
    in_str = None
    escape = False

    for i, ch in enumerate(array_text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in ("'", '"', "`"):
            in_str = ch
            continue
        if ch == "{":
            if depth == 0:
                obj_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and obj_start is not None:
                objects.append(array_text[obj_start : i + 1])
                obj_start = None
    return objects


def _extract_field(obj_text, field_name):
    m = re.search(
        rf'\b{re.escape(field_name)}\s*:\s*(?:"((?:\\"|[^"])*)"|\'((?:\\\'|[^\'])*)\')',
        obj_text,
    )
    if not m:
        return None
    raw = m.group(1) if m.group(1) is not None else m.group(2)
    return raw.replace('\\"', '"').replace("\\'", "'").strip()


def _extract_id(obj_text):
    m = re.search(r"\bid\s*:\s*(\d+)", obj_text)
    return int(m.group(1)) if m else None


def extract_audio_items(html_path):
    if not os.path.exists(html_path):
        raise FileNotFoundError(
            f"Không tìm thấy file '{html_path}'. Hãy đặt script này cùng thư mục với {html_path}!"
        )

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    items = [
        (
            "sample.mp3",
            "Guten Morgen, Chef! Was steht heute beim Mise en Place auf dem Plan?",
        )
    ]

    # 1.1 Trích xuất động VOCAB_DATA
    vocab_arr = _extract_js_array_block(content, "VOCAB_DATA")
    vocab_objs = _split_top_level_objects(vocab_arr)
    for obj in vocab_objs:
        vid = _extract_id(obj)
        de_word = _extract_field(obj, "de")
        ex_de = _extract_field(obj, "ex_de")
        if vid is not None and de_word:
            items.append((f"vocab_{vid}_word.mp3", de_word))
        if vid is not None and ex_de:
            items.append((f"vocab_{vid}_ex.mp3", ex_de))

    # 1.2 Trích xuất động PHRASES_DATA
    phrase_arr = _extract_js_array_block(content, "PHRASES_DATA")
    phrase_objs = _split_top_level_objects(phrase_arr)
    for obj in phrase_objs:
        pid = _extract_id(obj)
        de_phrase = _extract_field(obj, "de")
        if pid is not None and de_phrase:
            items.append((f"phrase_{pid}.mp3", de_phrase))

    # 1.3 Trích xuất động DIALOGUES_DATA
    dlg_arr = _extract_js_array_block(content, "DIALOGUES_DATA")
    dlg_objs = _split_top_level_objects(dlg_arr)
    total_dlg_lines = 0
    for obj in dlg_objs:
        dlg_id = _extract_id(obj)
        if dlg_id is not None:
            lines_arr = _extract_js_array_block(obj, "lines")
            line_objs = _split_top_level_objects(lines_arr)
            for line_idx, lobj in enumerate(line_objs, start=1):
                de_line = _extract_field(lobj, "de")
                if de_line:
                    items.append((f"dialog_{dlg_id}_line_{line_idx}.mp3", de_line))
                    total_dlg_lines += 1

    stats = {
        "vocab": len(vocab_objs),
        "phrases": len(phrase_objs),
        "dialogues": len(dlg_objs),
        "dialogue_lines": total_dlg_lines,
        "total_items": len(items),
    }
    return items, stats


# ==============================================================================
# 2. QUẢN LÝ MANIFEST & KIỂM TRA FILE HỢP LỆ
# ==============================================================================
def normalize_tts_text(text):
    # Loại bỏ phần chú thích trong ngoặc đơn như (-n), (-e), (-en), (-), (Pl.)
    no_parens = re.sub(r"\s*\([^)]*\)", "", text)
    return (
        no_parens.replace("Ø", "Durchmesser ")
        .replace("/", ", ")
        .replace("–", ", ")
        .strip()
    )


def text_hash(text):
    clean_text = normalize_tts_text(text)
    return hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:16]


def load_manifest():
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_manifest(manifest):
    os.makedirs(BASE_AUDIO_DIR, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def is_valid_file(filepath):
    return os.path.exists(filepath) and os.path.getsize(filepath) > 256


def get_pending_items(voice_key, out_dir, items, manifest):
    voice_manifest = manifest.get(voice_key, {})
    pending = []
    for filename, text in items:
        out_path = os.path.join(out_dir, filename)
        h = text_hash(text)
        if not is_valid_file(out_path) or voice_manifest.get(filename) != h:
            pending.append((filename, text, h))
    return pending


# ==============================================================================
# 3. ENGINE 1: PIPER THORSTEN OFFLINE TTS
# ==============================================================================
def ensure_piper_model(cfg):
    os.makedirs(PIPER_MODELS_DIR, exist_ok=True)
    onnx_path = os.path.join(PIPER_MODELS_DIR, cfg["model"])
    json_path = onnx_path + ".json"

    if not is_valid_file(onnx_path):
        print(f"  Đang tải mô hình giọng Đức {cfg['model']}...")
        urllib.request.urlretrieve(cfg["url_onnx"], onnx_path)
    if not is_valid_file(json_path):
        urllib.request.urlretrieve(cfg["url_json"], json_path)
    return onnx_path


def synthesize_piper_audio(voice, text, out_path):
    clean_text = normalize_tts_text(text)
    try:
        import lameenc

        has_lame = True
    except ImportError:
        has_lame = False

    try:
        if has_lame:
            pcm_chunks = []
            sample_rate = voice.config.sample_rate
            for ch in voice.synthesize(clean_text):
                pcm_chunks.append(ch.audio_int16_bytes)
                sample_rate = getattr(ch, "sample_rate", sample_rate)
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(128)
            encoder.set_in_sample_rate(sample_rate)
            encoder.set_channels(1)
            encoder.set_quality(2)
            mp3_bytes = encoder.encode(b"".join(pcm_chunks)) + encoder.flush()
            with open(out_path, "wb") as f:
                f.write(mp3_bytes)
        else:
            with wave.open(out_path, "wb") as wav_file:
                if hasattr(voice, "synthesize_wav"):
                    voice.synthesize_wav(clean_text, wav_file)
                else:
                    try:
                        chunks = list(voice.synthesize(clean_text))
                        if chunks:
                            first = chunks[0]
                            wav_file.setnchannels(getattr(first, "sample_channels", 1))
                            wav_file.setsampwidth(getattr(first, "sample_width", 2))
                            wav_file.setframerate(
                                getattr(first, "sample_rate", voice.config.sample_rate)
                            )
                            for ch in chunks:
                                wav_file.writeframes(ch.audio_int16_bytes)
                        else:
                            wav_file.setnchannels(1)
                            wav_file.setsampwidth(2)
                            wav_file.setframerate(voice.config.sample_rate)
                    except TypeError:
                        voice.synthesize(clean_text, wav_file)
    except Exception:
        if os.path.exists(out_path):
            os.remove(out_path)
        raise


def generate_piper_tts(items, manifest):
    try:
        from piper.voice import PiperVoice
    except ImportError:
        print("[!] Thiếu thư viện piper-tts. Cài đặt bằng: pip install piper-tts")
        return

    for folder_key, cfg in PIPER_VOICES.items():
        out_dir = os.path.join(BASE_AUDIO_DIR, folder_key)
        os.makedirs(out_dir, exist_ok=True)
        manifest.setdefault(folder_key, {})

        pending = get_pending_items(folder_key, out_dir, items, manifest)
        print(
            f"\n>>> Giọng [{cfg['name']}]: Cần tạo/cập nhật {len(pending)}/{len(items)} file tại '{out_dir}'..."
        )
        if not pending:
            continue

        model_path = ensure_piper_model(cfg)
        voice = PiperVoice.load(model_path)

        for idx, (filename, text, h) in enumerate(pending, start=1):
            out_path = os.path.join(out_dir, filename)
            synthesize_piper_audio(voice, text, out_path)
            manifest[folder_key][filename] = h
            if idx % 25 == 0 or idx == len(pending):
                save_manifest(manifest)
                print(f"  [{idx}/{len(pending)}] Đã tạo tới: {out_path}")


# ==============================================================================
# 4. ENGINE 2: MICROSOFT EDGE NEURAL TTS
# ==============================================================================
async def _edge_batch(folder_key, voice_short_name, out_dir, pending, manifest):
    import edge_tts

    sem = asyncio.Semaphore(6)
    completed = 0
    total = len(pending)

    async def synth_one(filename, text, h):
        nonlocal completed
        out_path = os.path.join(out_dir, filename)
        clean_text = normalize_tts_text(text)
        async with sem:
            for attempt in range(3):
                try:
                    comm = edge_tts.Communicate(clean_text, voice_short_name)
                    await comm.save(out_path)
                    manifest[folder_key][filename] = h
                    completed += 1
                    if completed % 25 == 0 or completed == total:
                        save_manifest(manifest)
                        print(f"  [{completed}/{total}] Đã tạo tới: {out_path}")
                    return
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(1.5 * (attempt + 1))

    await asyncio.gather(*(synth_one(fn, txt, h) for fn, txt, h in pending))


def generate_edge_tts(items, manifest):
    try:
        import edge_tts  # noqa: F401
    except ImportError:
        print("[!] Thiếu thư viện edge-tts. Cài đặt bằng: pip install edge-tts")
        return

    for folder_key, (label, voice_code) in EDGE_VOICES.items():
        out_dir = os.path.join(BASE_AUDIO_DIR, folder_key)
        os.makedirs(out_dir, exist_ok=True)
        manifest.setdefault(folder_key, {})

        pending = get_pending_items(folder_key, out_dir, items, manifest)
        print(
            f"\n>>> Giọng [{label}]: Cần tạo/cập nhật {len(pending)}/{len(items)} file tại '{out_dir}'..."
        )
        if not pending:
            continue
        asyncio.run(_edge_batch(folder_key, voice_code, out_dir, pending, manifest))


# ==============================================================================
# MAIN MENU
# ==============================================================================
if __name__ == "__main__":
    items, stats = extract_audio_items(HTML_SOURCE_FILE)
    manifest = load_manifest()

    print("=================================================================")
    print(" NGÀNH BẾP & PHỤ BẾP (KOCH & KÜCHENHILFE) — TRÍCH XUẤT ĐỘNG TỪ HTML")
    print(f" • Từ vựng: {stats['vocab']} từ ({stats['vocab'] * 2} file từ + ví dụ)")
    print(f" • Mẫu câu: {stats['phrases']} câu ({stats['phrases']} file)")
    print(
        f" • Hội thoại: {stats['dialogues']} đoạn ({stats['dialogue_lines']} lượt thoại)"
    )
    print(f" => TỔNG CỘNG: {stats['total_items']} file âm thanh / mỗi giọng")
    print(f" THƯ MỤC XUẤT ÂM THANH: {BASE_AUDIO_DIR}/<tên_giọng>/")
    print("=================================================================")
    print("1. Tạo 2 giọng Piper Thorsten Offline (Chuẩn Đức & Rostock)")
    print("2. Tạo 2 giọng Microsoft Edge Neural (Nữ Katja & Nam Conrad)")
    print("3. Tạo TẤT CẢ 4 GIỌNG (Piper + Edge — Tự bỏ qua file không đổi)")
    choice = input("\nNhập lựa chọn của bạn (1-3): ").strip()

    if choice == "1":
        generate_piper_tts(items, manifest)
    elif choice == "2":
        generate_edge_tts(items, manifest)
    elif choice == "3":
        generate_piper_tts(items, manifest)
        generate_edge_tts(items, manifest)
        print("\n[✓] Hoàn tất tạo toàn bộ âm thanh 4 giọng cho bep!")
    else:
        print("Lựa chọn không hợp lệ.")
