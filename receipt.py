# coding: utf-8
"""
演習記録（提出用）

capture.py と flann.py は、実行するたびに「どの機能を、どれくらいの時間、どんな設定で試したか」を
記録します。記録は receipts/record_学籍番号.txt の 1 つのファイルにまとめて保存され、
実行のたびに追記されます（1 分ごとと、終了（q）したとき、R キーを押したときに保存）。
提出するのはこのファイルだけです。

    python capture.py                      # 初回だけ学籍番号を聞かれます（2 回目からは不要）
    python capture.py --id 学籍番号         # 学籍番号を指定する（変更するとき）
    python capture.py --task 課題コード     # 課題コードを付けて記録する（指示があったとき）
    python capture.py --no-record          # 記録しない

記録に含まれるもの（画像は一切含みません）:
  ・学籍番号、課題コード（指定したとき）、日時
  ・この環境のインストール ID（初回の実行時に乱数で作られ、このフォルダと
    ユーザーの設定フォルダの 2 か所に保存されます）
  ・PC の識別情報をハッシュにしたもの（元の情報には戻せません）
  ・OS・Python・OpenCV の版、カメラの解像度、FPS
  ・機能・設定（←/→ で変える値、表示の切り替えなど）ごとのフレーム数と時間、キー操作の回数、
    処理中の数値の要約（入力画像の明るさ、二値化のしきい値、顔・人物・直線などが見つかったフレーム数など）

記録の形式（あとから機能が増えても読めるように）:
  ・各スクリプトは「試せる項目の一覧（カタログ）」を記録に書き込みます。項目は
    "proc:binary"（機能）、"param:quantize=4"（←/→ の値）、"set:houghline/sigma=2"（表示の切り替え）、
    "blur:median"（前処理）のような名前で、網羅の割合はこの一覧に対して計算します。
  ・実行ごとの記録（セッション）を並べて保存します。知らない項目や欄は読み飛ばせる形です。
  ・本体は圧縮して暗号化し、改ざんを見つけるための検査用の値を付けてあります。書き換えると無効になります。

注意: 同じ環境（フォルダのコピーや同じ PC）で複数の人の記録を作ると、提出物から分かります。
"""

import base64
import datetime
import hashlib
import hmac
import json
import os
import platform
import time
import uuid
import zlib

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RECEIPT_DIR = os.path.join(HERE, "receipts")
LOCAL_ID_FILE = os.path.join(HERE, ".install_id")
FORMAT_VERSION = 6

TRIED_SECONDS = 3.0         # 項目を「試した」とみなす時間 [秒]（一瞬通り過ぎたものは数えない）
UNIT_NEED = 4               # 1 つのパラメータ・切り替えで、いくつの値を試せば「試した」とするか（値がそれより少なければ全部）
AUTOSAVE_SECONDS = 60       # 自動で保存する間隔 [秒]
MAX_GAP = 1.0               # 1 フレームの時間の上限 [秒]（止まっていた時間を数えない）

# 記録を暗号化する鍵・検査用の値の鍵（秘密ではないが、記録を手で読んだり書き換えたりできないようにする）
_ENC_KEY = hashlib.sha256(b"cv-lecture-samples/record/enc/v6").digest()
_MAC_KEY = hashlib.sha256(b"cv-lecture-samples/record/mac/v6").digest()
_MAGIC = b"CVR6"
# PC の識別情報をハッシュにするときに混ぜる値
_MACHINE_SALT = b"cv-lecture-samples/machine/v1"

GROUP_NAMES = {"proc": "機能", "param": "パラメータ", "set": "表示の切り替え", "blur": "前処理", "flann": "特徴量・推定法"}


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def item_group(item):
    """項目名の種類（"proc:binary" → "proc"）"""
    return item.split(":", 1)[0]


def item_unit(item):
    """項目がどのパラメータ・切り替えの値か（"param:quantize=4" → "param:quantize"。値のない項目はそのもの）"""
    return item.split("=", 1)[0]


# ----------------------------------------------------------------------
# インストール ID と学籍番号
# ----------------------------------------------------------------------
def _user_id_file():
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "cv-lecture-samples", "install_id")


def _read_id(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("id"), str) and len(data["id"]) == 32:
            return data
    except (OSError, ValueError):
        pass
    return None


def _write_id(path, data):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return True
    except OSError:
        return False


def ensure_install_id():
    """インストール ID を用意する（なければ作る）。
    このフォルダとユーザーの設定フォルダの 2 か所に保存し、片方が消えたらもう片方から戻す。
    2 か所の ID が食い違う（例: 他人のフォルダをコピーした）ときは、両方をそのまま記録する。"""
    user_file = _user_id_file()
    local, user = _read_id(LOCAL_ID_FILE), _read_id(user_file)
    if local is None and user is None:
        data = {"id": uuid.uuid4().hex, "created": now_iso()}
        _write_id(LOCAL_ID_FILE, data)
        _write_id(user_file, data)
        local = user = data
        event = "created"
    elif local is None:
        _write_id(LOCAL_ID_FILE, user)
        local = user
        event = "restored_local"
    elif user is None:
        _write_id(user_file, local)
        user = local
        event = "restored_user"
    else:
        event = "ok" if local["id"] == user["id"] else "mismatch"
    return {"local_id": local["id"], "local_created": local.get("created"),
            "user_id": user["id"], "user_created": user.get("created"), "event": event}


def normalize(text):
    return "".join(str(text).split()).upper()


def valid_student_id(sid):
    return bool(sid) and len(sid) <= 20 and all(c.isalnum() or c in "-_" for c in sid)


def ask_student_id():
    while True:
        sid = normalize(input("学籍番号を入力してください: "))
        if valid_student_id(sid):
            return sid
        print("英数字で入力してください。")


def get_student_id(given=None):
    """学籍番号を決める。given（--id）があればそれを使って保存し、なければ保存してある番号、
    どちらもなければ入力を求める（このフォルダの .install_id に保存するので、2 回目からは聞かない）。
    入力できない（ターミナルがない）ときは None（記録しない）"""
    local = _read_id(LOCAL_ID_FILE) or {}
    if given:
        sid = normalize(given)
        if not valid_student_id(sid):
            print("学籍番号は英数字で指定してください: %s" % given)
            return get_student_id(None)
    elif valid_student_id(local.get("student_id", "")):
        return local["student_id"]
    else:
        try:
            sid = ask_student_id()
        except (EOFError, KeyboardInterrupt):
            print("\n学籍番号を入力できないので、演習記録をとらずに続けます（--id 学籍番号 で指定できます）。")
            return None
    if local.get("id"):
        local["student_id"] = sid
        _write_id(LOCAL_ID_FILE, local)
    return sid


def machine_hash():
    """PC の識別情報（ネットワーク機器の固有番号・ホスト名・ユーザー名）のハッシュ。元には戻せない"""
    node = uuid.getnode()
    reliable = not (node >> 40) & 1          # 固有番号が取れず乱数になったときは 1 が立つ
    try:
        user = os.getlogin()
    except OSError:
        user = os.environ.get("USER") or os.environ.get("USERNAME") or ""
    raw = ("%012x|%s|%s" % (node, platform.node(), user)).encode("utf-8")
    return hashlib.sha256(_MACHINE_SALT + raw).hexdigest()[:16], bool(reliable)


# ----------------------------------------------------------------------
# 記録ファイルの読み書き（圧縮 → 暗号化 → 検査用の値 → 文字列）
# ----------------------------------------------------------------------
def _keystream(nonce, n):
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hashlib.sha256(_ENC_KEY + nonce + i.to_bytes(4, "big")).digest()
        i += 1
    return np.frombuffer(bytes(out[:n]), np.uint8)


def encode_body(data):
    """記録（辞書）を文字列にする。戻り値: (文字列, 確認コード)"""
    raw = zlib.compress(json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 9)
    nonce = os.urandom(16)
    cipher = (np.frombuffer(raw, np.uint8) ^ _keystream(nonce, len(raw))).tobytes()
    mac = hmac.new(_MAC_KEY, _MAGIC + nonce + cipher, hashlib.sha256).digest()
    blob = base64.b64encode(_MAGIC + nonce + mac + cipher).decode()
    return blob, check_code(mac)


def check_code(mac):
    h = mac.hex().upper()
    return "%s-%s-%s" % (h[0:4], h[4:8], h[8:12])


def decode_receipt(text):
    """記録ファイルの本文から (データ, 検査用の値が正しいか, 確認コード) を返す。読めなければ例外"""
    inside = text.split("-----BEGIN RECORD-----", 1)[1].split("-----END RECORD-----", 1)[0]
    blob = base64.b64decode("".join(inside.split()))
    if blob[:4] != _MAGIC:
        raise ValueError("unknown format")
    nonce, mac, cipher = blob[4:20], blob[20:52], blob[52:]
    ok = hmac.compare_digest(mac, hmac.new(_MAC_KEY, _MAGIC + nonce + cipher, hashlib.sha256).digest())
    raw = (np.frombuffer(cipher, np.uint8) ^ _keystream(nonce, len(cipher))).tobytes()
    try:
        data = json.loads(zlib.decompress(raw).decode("utf-8"))
    except (zlib.error, ValueError):
        if ok:
            raise
        data = None                             # 書き換えられていて中身も読めない
    return data, ok, check_code(mac)


def record_path(student_id):
    return os.path.join(RECEIPT_DIR, "record_%s.txt" % student_id)


def load_record(student_id):
    """保存してある記録を読む。(データ or None, 状態) 状態: "none" / "ok" / "broken" """
    path = record_path(student_id)
    if not os.path.exists(path):
        return None, "none"
    try:
        with open(path, encoding="utf-8") as f:
            data, ok, _ = decode_receipt(f.read())
        if ok and data and data.get("student_id") == student_id and isinstance(data.get("sessions"), list):
            return data, "ok"
    except Exception:
        pass
    return None, "broken"


# ----------------------------------------------------------------------
# 集計
# ----------------------------------------------------------------------
def merge_items(sessions, script=None):
    """セッションの項目ごとの [フレーム数, 秒数] を合計する（script を指定するとそのスクリプトのみ）"""
    total = {}
    for s in sessions:
        if script and s.get("script") != script:
            continue
        for item, (frames, seconds) in s.get("items", {}).items():
            t = total.setdefault(item, [0, 0.0])
            t[0] += frames
            t[1] += seconds
    return total


def unit_progress(catalog, items, tried_seconds=TRIED_SECONDS, need=UNIT_NEED):
    """パラメータ・切り替えごとに (試した値の数, 必要な数) を返す。
    1 つの値を tried_seconds 秒以上使ったら、その値を試したとする。
    必要な数は、値の数と need の小さい方（機能・前処理のように値のないものは 1）"""
    values = {}
    for item in catalog:
        values.setdefault(item_unit(item), []).append(item)
    return {u: (sum(1 for it in its if items.get(it, [0, 0.0])[1] >= tried_seconds), min(len(its), need))
            for u, its in values.items()}


def coverage(catalog, items, tried_seconds=TRIED_SECONDS, need=UNIT_NEED):
    """種類（proc / param / set / blur / flann）ごとに (試し終えた数, 全体の数) を返す。
    数えるのはパラメータ・切り替えの単位（例: 「量子化のビット数」で 1 つ）。items: 項目 → [フレーム数, 秒数]"""
    result = {}
    for unit, (done, req) in unit_progress(catalog, items, tried_seconds, need).items():
        t = result.setdefault(item_group(unit), [0, 0])
        t[1] += 1
        if done >= req:
            t[0] += 1
    return result


def sort_groups(cov):
    order = list(GROUP_NAMES)
    return sorted(cov.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else len(order))


def header_lines(data, code):
    """記録ファイルの先頭の、人が読める部分（教員用の確認スクリプトは本体と一致するかを確かめる）"""
    lines = ["# 演習記録（このファイルを提出してください。書き換えると無効になります）",
             "# 学籍番号: %s   更新: %s   確認コード: %s" % (data["student_id"], data["updated"], code)]
    sessions = data["sessions"]
    for script in sorted(data["catalogs"]):
        sec = sum(s.get("seconds", 0.0) for s in sessions if s.get("script") == script)
        n = sum(1 for s in sessions if s.get("script") == script)
        cov = coverage(data["catalogs"][script], merge_items(sessions, script))
        parts = ["%s %d/%d" % (GROUP_NAMES.get(g, g), a, b) for g, (a, b) in sort_groups(cov)]
        lines.append("# %s: %d 回 %s  試し終えた項目: %s" % (script, n, format_duration(sec), "  ".join(parts)))
    return lines


def format_duration(seconds):
    m = int(seconds // 60)
    return "%d:%02d" % (m // 60, m % 60)


# ----------------------------------------------------------------------
# 記録
# ----------------------------------------------------------------------
class Recorder:
    """機能・設定ごとの利用時間と、キー操作の回数を記録する。
    catalog: このスクリプトで試せる項目の名前のリスト（網羅の割合の計算に使う）"""

    def __init__(self, script, student_id, catalog, task_code="", source="camera", frame_size=(0, 0)):
        self.script = script                    # 記録をとったスクリプト（capture / flann）
        self.student_id = student_id
        self.task_code = normalize(task_code) if task_code else ""
        self.catalog = list(catalog)
        self.session_id = uuid.uuid4().hex
        self.source = source                    # "camera" または "file"
        self.frame_size = list(frame_size)      # (幅, 高さ)
        self.install = ensure_install_id()
        self.machine, self.machine_reliable = machine_hash()
        self.started = time.time()
        self.started_iso = now_iso()
        self.prev_t = None
        self.prev_proc = None
        self.last_save = time.time()
        self.frames = 0
        self.seconds = 0.0
        self.help_seconds = 0.0                 # ヘルプを表示していた時間（機能を試した時間には数えない）
        self.switches = 0                       # 処理を切り替えた回数（合計）
        self.modes = {}                         # 処理名 → 利用状況
        self.items = {}                         # 項目名 → [フレーム数, 秒数]
        self.keys = {}                          # "キー@処理名" → 押した回数
        self.saves = 0
        prev, self.load_state = load_record(student_id)
        if self.load_state == "broken":         # 壊れた記録は残しておき、新しく始める
            os.replace(record_path(student_id),
                       record_path(student_id) + ".broken-%s" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.prev_sessions = prev["sessions"] if prev else []
        self.broken_count = (prev or {}).get("broken_count", 0) + (1 if self.load_state == "broken" else 0)
        self.prev_items = merge_items(self.prev_sessions, script)

    # --- 毎フレーム ---
    def on_key(self, key, proc):
        """キーが押されたときに呼ぶ。proc は押した時点で実行中の処理"""
        if key is None:
            return
        name = {chr(27): "ESC", " ": "SPACE"}.get(key, key)
        k = "%s@%s" % (name, proc)
        self.keys[k] = self.keys.get(k, 0) + 1

    def record(self, proc, img, st=None, param_value=None, values=None, items=(), paused=False):
        """1 フレームごとに呼ぶ。
        proc: 処理名、img: 処理前の入力画像、st: 状態（st.found = 検出系の処理で見つけた数）、
        param_value: ←/→ で変えるパラメータの値、values: {名前: 数値}（平均・ばらつき・最小・最大を記録）、
        items: このフレームで使っている項目の名前（カタログの名前）、
        paused: True ならヘルプの表示中などで、機能を試した時間には数えない（help_seconds にだけ足す）"""
        t = time.time()
        dt = 0.0 if self.prev_t is None else min(t - self.prev_t, MAX_GAP)
        self.prev_t = t
        if paused:
            self.help_seconds += dt
            if t - self.last_save >= AUTOSAVE_SECONDS:
                self.save()
            return
        self.frames += 1
        self.seconds += dt
        for item in items:
            it = self.items.setdefault(item, [0, 0.0])
            it[0] += 1
            it[1] += dt
        m = self.modes.get(proc)
        if m is None:
            m = self.modes[proc] = {"frames": 0, "seconds": 0.0, "entries": 0, "bright_sum": 0.0,
                                    "bright_sq": 0.0, "bright_min": 255.0, "bright_max": 0.0}
        if proc != self.prev_proc:
            m["entries"] += 1
            if self.prev_proc is not None:
                self.switches += 1
            self.prev_proc = proc
        m["frames"] += 1
        m["seconds"] += dt
        b = float(cv2.cvtColor(cv2.resize(img, (64, 48)), cv2.COLOR_BGR2GRAY).mean())
        m["bright_sum"] += b
        m["bright_sq"] += b * b
        m["bright_min"] = min(m["bright_min"], b)
        m["bright_max"] = max(m["bright_max"], b)
        for name, v in (values or {}).items():      # 数値の要約（flann.py の対応の数など）
            if v is None:
                continue
            s = m.setdefault("stats", {}).setdefault(name, [0, 0.0, 0.0, float(v), float(v)])
            s[0] += 1
            s[1] += float(v)
            s[2] += float(v) ** 2
            s[3] = min(s[3], float(v))
            s[4] = max(s[4], float(v))
        if param_value is not None:                 # パラメータの値ごとのフレーム数
            pv = m.setdefault("params", {})
            pv[str(param_value)] = pv.get(str(param_value), 0) + 1
        found = getattr(st, "found", None)          # 検出系の処理（顔・人物・骨格・直線・円）で見つけた数
        if found is not None:
            m["detect_frames"] = m.get("detect_frames", 0) + 1
            m["found_frames"] = m.get("found_frames", 0) + (1 if found > 0 else 0)
            m["found_total"] = m.get("found_total", 0) + int(found)
        if proc == "binary" and st is not None:
            otsu = int(getattr(st, "bin_last_otsu", -1))
            m.setdefault("otsu", {"min": 255, "max": 0, "sum": 0})
            m["otsu"]["min"] = min(m["otsu"]["min"], otsu)
            m["otsu"]["max"] = max(m["otsu"]["max"], otsu)
            m["otsu"]["sum"] += otsu
            manual = getattr(st, "bin_manual", None)
            if manual is not None:
                held = m.setdefault("manual", {})
                held[str(manual)] = held.get(str(manual), 0) + 1
        if t - self.last_save >= AUTOSAVE_SECONDS:
            self.save()

    # --- 網羅の状況（これまでの実行も含む） ---
    def total_items(self):
        total = {k: list(v) for k, v in self.prev_items.items()}
        for item, (frames, seconds) in self.items.items():
            t = total.setdefault(item, [0, 0.0])
            t[0] += frames
            t[1] += seconds
        return total

    def tried(self):
        """これまでに試した項目の集合（TRIED_SECONDS 秒以上）"""
        return {k for k, (_, s) in self.total_items().items() if s >= TRIED_SECONDS}

    def coverage(self):
        return coverage(self.catalog, self.total_items())

    def progress(self):
        """パラメータ・切り替えごとの (試した値の数, 必要な数)"""
        return unit_progress(self.catalog, self.total_items())

    def total_seconds(self):
        return self.seconds + sum(s.get("seconds", 0.0) for s in self.prev_sessions if s.get("script") == self.script)

    # --- 保存 ---
    def session(self):
        modes = {}
        for proc, m in self.modes.items():
            n = max(m["frames"], 1)
            mean = m["bright_sum"] / n
            entry = {"frames": m["frames"], "seconds": round(m["seconds"], 2), "entries": m["entries"],
                     "bright_mean": round(mean, 2),
                     "bright_std": round(max(m["bright_sq"] / n - mean * mean, 0.0) ** 0.5, 2),
                     "bright_min": round(m["bright_min"], 2), "bright_max": round(m["bright_max"], 2)}
            if "otsu" in m:
                entry["otsu"] = {"min": m["otsu"]["min"], "max": m["otsu"]["max"],
                                 "mean": round(m["otsu"]["sum"] / n, 2)}
            for k in ("manual", "detect_frames", "found_frames", "found_total", "params"):
                if k in m:
                    entry[k] = m[k]
            if "stats" in m:
                entry["stats"] = {}
                for name, (cnt, sm, sq, mn, mx) in m["stats"].items():
                    mean = sm / cnt
                    entry["stats"][name] = {"n": cnt, "mean": round(mean, 3), "min": round(mn, 3), "max": round(mx, 3),
                                            "std": round(max(sq / cnt - mean * mean, 0.0) ** 0.5, 3)}
            modes[proc] = entry
        elapsed = time.time() - self.started
        return {
            "id": self.session_id,
            "script": self.script,
            "task_code": self.task_code,
            "started": self.started_iso,
            "updated": now_iso(),
            "install": self.install,
            "machine": {"hash": self.machine, "reliable": self.machine_reliable},
            "env": {"os": platform.platform(), "python": platform.python_version(),
                    "opencv": cv2.__version__, "numpy": np.__version__},
            "source": self.source, "frame_size": self.frame_size,
            "frames": self.frames, "seconds": round(self.seconds, 2), "elapsed": round(elapsed, 2),
            "help_seconds": round(self.help_seconds, 2),
            "fps": round(self.frames / max(elapsed, 1e-6), 2), "switches": self.switches,
            "saves": self.saves + 1,
            "modes": modes,
            "items": {k: [v[0], round(v[1], 2)] for k, v in self.items.items()},
            "keys": self.keys,
        }

    def save(self):
        """記録ファイルを保存し、(ファイルのパス, 確認コード) を返す。
        保存の直前に読み直すので、capture.py と flann.py を同時に動かしていても両方の記録が残る"""
        self.last_save = time.time()
        if self.frames == 0:
            return None, None
        prev, state = load_record(self.student_id)
        sessions = [s for s in (prev["sessions"] if prev else self.prev_sessions) if s.get("id") != self.session_id]
        sessions.append(self.session())
        catalogs = dict((prev or {}).get("catalogs", {}))
        catalogs[self.script] = self.catalog
        data = {"format": FORMAT_VERSION, "student_id": self.student_id, "updated": now_iso(),
                "tried_seconds": TRIED_SECONDS, "unit_need": UNIT_NEED, "catalogs": catalogs, "sessions": sessions,
                "broken_count": self.broken_count}
        blob, code = encode_body(data)
        self.saves += 1

        lines = header_lines(data, code) + ["-----BEGIN RECORD-----"]
        lines += [blob[i:i + 76] for i in range(0, len(blob), 76)]
        lines += ["-----END RECORD-----", ""]
        os.makedirs(RECEIPT_DIR, exist_ok=True)
        path = record_path(self.student_id)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        os.replace(tmp, path)                   # 書き込みの途中で止まっても前の記録が壊れないように
        return path, code
