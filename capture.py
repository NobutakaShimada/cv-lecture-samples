# coding: utf-8
"""
カメラキャプチャ＆基本的な画像処理サンプル

Webカメラから画像を1フレームずつ取り込み、キー操作で選んだ処理を施して表示します。

使い方:
    python capture.py            # カメラ番号 0 を使う
    python capture.py 1          # カメラ番号 1 を使う
    python capture.py movie.mp4  # カメラの代わりに動画ファイルを使う（繰り返し再生）
    python capture.py --id 学籍番号        # 学籍番号を変える（初回は起動時に聞かれる）
    python capture.py --no-record        # 演習記録をとらない

実行中の操作は演習記録として receipts/record_学籍番号.txt に保存されます（receipt.py を参照）。

キー操作は起動時にターミナルへ表示されます（README.md も参照）。

自分で処理を追加したいときは、下の「proc_custom」関数を書き換えて
キー 'u' を押してください。
"""

import argparse
import atexit
import os
import sys
import time
import random

import cv2
import numpy as np


# ----------------------------------------------------------------------
# 設定
# ----------------------------------------------------------------------
PROC_SCALE = 0.5        # 処理速度のため、入力画像をこの倍率に縮小してから処理する
WINDOW_NAME = "OpenCV Capture"
SHOW_HELP_AT_START = True   # 起動時にウィンドウにヘルプを重ねて表示する（? キーで ON/OFF）
HELP_DIM = 0.3              # ヘルプ表示中の背景の明るさ（0〜1、小さいほど暗い）

# 学習済みモデルの置き場所（このスクリプトと同じフォルダの models/）
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
HAAR_DIR = os.path.join(MODEL_DIR, "haarcascades")
YUNET_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")


# ----------------------------------------------------------------------
# 状態（パラメータ等）をまとめて持つクラス
# ----------------------------------------------------------------------
class State:
    def __init__(self, proc_shape):
        h, w = proc_shape[:2]

        # 実行中の処理名
        self.proc = "color"

        # 前処理（ぼかし）の種類とパラメータ
        self.blur_proc = "gaussian"
        self.reset_param()

        # 前フレーム（オプティカルフロー用）
        self.prvs = None

        # HSV 表示用
        self.hsv = np.zeros((h, w, 3), np.uint8)
        self.hsv[..., 1] = 255

        # ラベリング用のランダムカラー表（HSV）
        colors = [[0, 0, 0]]
        for _ in range(1, 256):
            colors.append([random.randint(0, 180), random.randint(120, 255), 255])
        self.colors_hsv = np.array(colors, dtype=np.uint8)

        # 顔検出器は必要になったときに読み込む（load_haar / load_yunet）
        self.haar = None          # Haar-like 特徴カスケード（古典的手法）
        self.yunet = None         # YuNet（CNN による顔検出）
        self.pose = None          # MediaPipe Pose（人体の骨格推定）
        self.profile_face = True
        self.found = None         # 検出系の処理が見つけた数（演習記録用。検出しない処理では None）

        # Lucas-Kanade 追跡用
        self.lk_fnum = 1000
        # Shi-Tomasi コーナー検出器のためのパラメータ
        self.feature_params = dict(maxCorners=self.lk_fnum,
                                   qualityLevel=0.001,
                                   minDistance=7,
                                   blockSize=7)
        # Lucas-Kanade 法によるオプティカルフローのためのパラメータ
        self.lk_params = dict(winSize=(15, 15),
                              maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.lk_color = np.random.randint(0, 255, (self.lk_fnum, 3))
        self.lk_reset = True
        self.lk_points = None
        self.track_list = []

        # SIFT / CLAHE は必要になったときに作る
        self.sift = None
        self.clahe = None

        # ←/→ で変える処理ごとのパラメータ（PARAMS の値の候補の何番目か）と、ヒストグラム表示
        self.param_index = {}
        self.show_hist = True
        self.hog = None           # HOG 特徴 + SVM の人物検出器
        self.hog_view = False     # P で画像全体の HOG 特徴を表示する（o キーで切り替え）
        self.tried = None         # 演習記録で「試した」項目の集合（ヘルプで未体験の機能に印を付ける。記録しないときは None）
        self.progress = {}        # 演習記録: パラメータごとの (試した値の数, 必要な数)
        self.hough_sigma = 0      # Hough 変換のぼかし投票の広がり σ（0 は 1 つのセルにだけ 1 票。w キーで切り替え）

        # 二値化のしきい値: None なら大津の方法で毎フレーム自動、数値なら ←/→ で決めた手動の値
        self.bin_manual = None
        self.bin_last_otsu = 128      # 直近のフレームの大津のしきい値（手動に切り替えるときの起点）

    # --- ぼかしパラメータ（↑↓キーで変更、ESCでリセット）---
    def increase_param(self):
        self.gaussian_scale = min(self.gaussian_scale + 6, 149)
        self.median_scale = min(self.median_scale + 6, 199)
        self.bilateral_scale = min(self.bilateral_scale + 5, 30)

    def decrease_param(self):
        self.gaussian_scale = max(self.gaussian_scale - 6, 1)
        self.median_scale = max(self.median_scale - 6, 1)
        self.bilateral_scale = max(self.bilateral_scale - 5, 1)

    def reset_param(self):
        # ガウシアン・メディアンのカーネルサイズは奇数である必要がある
        self.gaussian_scale = 1
        self.median_scale = 1
        self.bilateral_scale = 1


# ----------------------------------------------------------------------
# 顔検出器の読み込み
# ----------------------------------------------------------------------
def load_haar(st):
    """Haar-like 特徴カスケードを読み込む。使えなければ False を返す"""
    if st.haar is not None:
        return st.haar is not False
    if not hasattr(cv2, "CascadeClassifier"):
        # OpenCV 5 では CascadeClassifier は contrib 版にのみ含まれる
        print("この OpenCV (%s) には CascadeClassifier がありません。" % cv2.__version__)
        print("opencv-contrib-python を入れてください（README.md 参照）。")
        st.haar = False
        return False
    names = {"face": "haarcascade_frontalface_default.xml",
             "eye": "haarcascade_eye_tree_eyeglasses.xml",
             "body": "haarcascade_upperbody.xml",
             "profile": "haarcascade_profileface.xml"}
    st.haar = {}
    for key, fname in names.items():
        c = cv2.CascadeClassifier(os.path.join(HAAR_DIR, fname))
        if c.empty():
            print("カスケードファイルを読み込めません: %s" % os.path.join(HAAR_DIR, fname))
            st.haar = False
            return False
        st.haar[key] = c
    return True


def load_yunet(st, size):
    """YuNet 顔検出器を読み込む。使えなければ False を返す"""
    if st.yunet is None:
        if not hasattr(cv2, "FaceDetectorYN"):
            print("この OpenCV (%s) には FaceDetectorYN がありません（4.8 以降が必要）。" % cv2.__version__)
            st.yunet = False
        elif not os.path.exists(YUNET_MODEL):
            print("YuNet のモデルファイルがありません: %s" % YUNET_MODEL)
            st.yunet = False
        else:
            # 引数: モデル, 設定(不要), 入力サイズ, スコア閾値, NMS閾値, 最大候補数
            st.yunet = cv2.FaceDetectorYN.create(YUNET_MODEL, "", size, 0.7, 0.3, 5000)
    if st.yunet is False:
        return False
    st.yunet.setInputSize(size)
    return True


def show_unavailable(img, msg):
    result = img.copy()
    cv2.putText(result, msg, (10, result.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 255), 2)
    return result


# ----------------------------------------------------------------------
# 前処理（ぼかし）
# ----------------------------------------------------------------------
def blur(img, st):
    if st.blur_proc == "gaussian":
        k = st.gaussian_scale
        return cv2.GaussianBlur(img, (k, k), 0)
    elif st.blur_proc == "median":
        return cv2.medianBlur(img, st.median_scale)
    elif st.blur_proc == "bilateral":
        return cv2.bilateralFilter(img, st.bilateral_scale, 75, 75)
    return img


# ----------------------------------------------------------------------
# 各種画像処理
#   引数 img: BGR カラー画像 (uint8)、st: State
#   戻り値: 表示する画像（カラーでもグレースケールでもよい）
# ----------------------------------------------------------------------
def proc_color(img, st):
    return img


def proc_gray(img, st):
    # グレースケール = 輝度 Y = 0.299R + 0.587G + 0.114B（YIQ や YUV の Y と同じ）
    # 人の目は緑に最も敏感で青に鈍いので、緑の重みが大きい
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return with_histogram(gray, st)


def with_histogram(gray, st):
    """グレースケール画像に、そのヒストグラムを重ねる（i キーで ON/OFF）"""
    if not st.show_hist:
        return gray
    result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    draw_histogram(result, gray)
    return result


def _sobel(img, dx, dy, ksize=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)


def proc_sobelx(img, st):
    dx = _sobel(img, 1, 0)
    return np.clip(np.abs(dx * 10), 0, 255).astype(np.uint8)


def proc_sobely(img, st):
    dy = _sobel(img, 0, 1)
    return np.clip(np.abs(dy * 10), 0, 255).astype(np.uint8)


def proc_sobel(img, st):
    dx = _sobel(img, 1, 0)
    dy = _sobel(img, 0, 1)
    mag = np.sqrt((dx * dx + dy * dy) / 2) * 10
    return np.clip(mag, 0, 255).astype(np.uint8)


def proc_angle(img, st):
    # 勾配の向きを色相(H)、強さを明度(V)で表示
    dx = _sobel(img, 1, 0, ksize=3)
    dy = _sobel(img, 0, 1, ksize=3)
    ang = np.mod(np.arctan2(dy, dx), 2 * np.pi)          # 0 〜 2π
    hsv = np.empty(img.shape, np.uint8)
    hsv[..., 0] = (ang / (2 * np.pi) * 180).astype(np.uint8)  # OpenCV の H は 0〜179
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(np.sqrt((dx * dx + dy * dy) / 2) * 10, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def proc_hue(img, st):
    # 色相（H）だけを残し、彩度と明度を最大にして表示
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = 255
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def proc_saturation(img, st):
    # 彩度（S）をグレースケールで表示: 鮮やかな色ほど白く、白・黒・灰色ほど黒い
    # S = (max(R,G,B) - min(R,G,B)) / max(R,G,B)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 1]


def proc_value(img, st):
    # 明度（V）をグレースケールで表示: V = max(R,G,B)
    # グレースケール（輝度 Y）と違い、純粋な赤・緑・青はどれも最大（白）になる
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return hsv[:, :, 2]


BIN_STEP = 2       # ←/→ 1 回で動かすしきい値の量


def proc_binary(img, st):
    # 二値化: ふだんは大津の方法でしきい値を自動で決める。
    # ←/→ を押すと手動になり、画像が変わってもしきい値はその値に固定される（ESC で自動に戻る）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    st.bin_last_otsu = int(otsu)
    thresh = int(otsu) if st.bin_manual is None else st.bin_manual
    _, binimg = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)   # thresh より大きい画素が白
    result = cv2.cvtColor(binimg, cv2.COLOR_GRAY2BGR)
    if st.show_hist:
        draw_histogram(result, gray, int(otsu), thresh, manual=st.bin_manual is not None)
    return result


def shift_threshold(st, step):
    """←/→ でしきい値を動かす。自動のときは、いまの大津の値を起点に手動に切り替える"""
    base = st.bin_last_otsu if st.bin_manual is None else st.bin_manual
    st.bin_manual = int(np.clip(base + step, 0, 255))


def between_class_variance(hist):
    """大津の方法の評価値: しきい値 t ごとのクラス間分散 σ_B²(t)。これが最大の t が大津のしきい値。
    ω(t): t 以下の画素の割合, μ(t): t 以下の画素の (値 x 割合) の和, μ_T: 全体の平均として
    σ_B²(t) = (μ_T ω(t) - μ(t))² / (ω(t) (1 - ω(t)))"""
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(np.arange(256) * p)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return np.nan_to_num(var)


def draw_histogram(img, gray, otsu=None, thresh=None, manual=False):
    """画像の右下にヒストグラムを重ねて表示する（横軸: 左が 0=黒, 右が 255=白）。
    二値化（thresh を指定）のとき: 灰色の棒: 黒になる画素（しきい値以下）, 白の棒: 白になる画素,
    水色の曲線: 大津の方法のクラス間分散, 緑の線: 大津のしきい値, 赤の線: 実際のしきい値。
    白飛び・黒つぶれで 0 や 255 に画素が集中すると他の棒が見えなくなるので、
    棒の高さは両端を除いたビンの最大に合わせ、それを超える両端はマゼンタで示す"""
    H, W = img.shape[:2]
    pw, ph = int(W * 0.45), int(H * 0.40)
    x0, y0 = W - pw - 8, H - ph - 8
    panel = img[y0:y0 + ph, x0:x0 + pw]
    panel[:] = (panel * 0.25).astype(np.uint8)                   # 背景を暗くする
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, max(ph / 300.0, 0.33)
    top, bottom = y0 + int(ph * 0.18), y0 + ph - int(ph * 0.12)  # 棒グラフを描く範囲
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    xs = x0 + 4 + np.arange(256) * (pw - 8) / 255.0
    ref = max(hist[1:255].max(), 1)                              # 両端を除いた最大
    heights = np.minimum(hist / ref, 1.0) * (bottom - top)
    for i in range(256):
        if i in (0, 255) and hist[i] > ref:
            color = (255, 0, 255)                                 # 飽和（白飛び・黒つぶれ）
        elif thresh is None:
            color = (200, 200, 200)
        else:
            color = (110, 110, 110) if i <= thresh else (235, 235, 235)
        cv2.line(img, (int(xs[i]), bottom), (int(xs[i]), int(bottom - heights[i])), color, 1)
    if thresh is not None:
        var = between_class_variance(hist)
        curve = np.stack([xs, bottom - var / max(var.max(), 1e-9) * (bottom - top)], axis=1)
        cv2.polylines(img, [np.round(curve).astype(np.int32)], False, (255, 255, 0), 1, cv2.LINE_AA)
        cv2.line(img, (int(xs[otsu]), top), (int(xs[otsu]), bottom), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(img, (int(xs[thresh]), top), (int(xs[thresh]), bottom), (0, 0, 255), 2, cv2.LINE_AA)
        mode = "manual, ESC: auto" if manual else "auto = Otsu, <-/->: manual"
        title = "Otsu %d  thresh %d (%s)" % (otsu, thresh, mode)
    else:
        title = "histogram  (mean %.1f)  i: hide" % float(gray.mean())
    # 文字: タイトル、横軸の目盛り、飽和した画素の割合
    cv2.putText(img, title, (x0 + 6, y0 + int(ph * 0.11)), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
    ty = y0 + ph - 4
    cv2.putText(img, "0", (x0 + 4, ty), font, scale, (200, 200, 200), 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize("255", font, scale, 1)
    cv2.putText(img, "255", (x0 + pw - tw - 4, ty), font, scale, (200, 200, 200), 1, cv2.LINE_AA)
    total = hist.sum()
    (w0, _), _ = cv2.getTextSize("0", font, scale, 1)
    for i in (0, 255):
        if hist[i] > ref:
            label = "%d: %.1f%%" % (i, hist[i] / total * 100)
            (lw, _), _ = cv2.getTextSize(label, font, scale, 1)
            lx = x0 + 4 + w0 + 10 if i == 0 else x0 + pw - tw - 14 - lw   # 目盛りの内側に置く
            cv2.putText(img, label, (lx, ty), font, scale, (255, 0, 255), 1, cv2.LINE_AA)


def proc_sift(img, st):
    if st.sift is None:
        st.sift = cv2.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp = st.sift.detect(gray, None)
    return cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


def proc_denseflow(img, st):
    # Farneback 法による密なオプティカルフロー
    nxt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if st.prvs is None or st.prvs.shape != nxt.shape:
        st.prvs = nxt
    flow = cv2.calcOpticalFlowFarneback(st.prvs, nxt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # 向きを色相、大きさを明度で表示
    st.hsv[..., 0] = ang * 180 / np.pi / 2
    st.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(st.hsv, cv2.COLOR_HSV2BGR)
    result = cv2.addWeighted(img, 0.5, rgb, 0.5, 0)
    # 間引いた位置に矢印（線分）を描く
    for i in range(0, result.shape[0], 15):
        for j in range(0, result.shape[1], 15):
            fx, fy = flow[i, j]
            cv2.line(result, (j, i), (int(j + fx * 3), int(i + fy * 3)), (255, 255, 255), 1)
    st.prvs = nxt
    return result


def proc_labeling(img, st):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, label = cv2.connectedComponents(binimg)
    return cv2.cvtColor(st.colors_hsv[label % 256], cv2.COLOR_HSV2BGR)


def proc_labeling2(img, st):
    # ラベリング＋図形の特徴: 外接矩形、面積 S、重心（×）、モーメントから求めた主軸（楕円と線）
    # ←/→ で表示する領域の最小面積を変える
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    num, label, stats, centroids = cv2.connectedComponentsWithStats(binimg)
    result = cv2.cvtColor(st.colors_hsv[label % 256], cv2.COLOR_HSV2BGR)
    min_area = param(st, "labeling2")
    font = cv2.FONT_HERSHEY_PLAIN
    # 画像の半分を超える大きな領域は背景とみなして除く
    big = [i for i in range(1, num) if min_area <= stats[i][4] <= label.size // 2]
    for i in sorted(big, key=lambda i: -stats[i][4])[:30]:        # 大きい順に 30 個まで
        x, y, w, h, area = stats[i]
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255))
        cx, cy, theta, major, minor = principal_axes(label[y:y + h, x:x + w] == i)
        cx, cy = cx + x, cy + y
        cv2.ellipse(result, (int(cx), int(cy)), (int(major), int(minor)), np.degrees(theta), 0, 360,
                    (255, 255, 255), 1, cv2.LINE_AA)
        dx, dy = major * np.cos(theta), major * np.sin(theta)
        cv2.line(result, (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy)), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(result, (int(cx), int(cy)), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
        cv2.putText(result, "S:%d" % area, (x, y + h + 13), font, 1, (0, 255, 255))
        cv2.putText(result, "deg:%.0f e:%.1f" % (np.degrees(theta), major / max(minor, 1e-6)),
                    (x, y + h + 26), font, 1, (0, 255, 255))
    return result


def principal_axes(mask):
    """領域（True の画素）のモーメントから、重心・主軸の向き・主軸方向の広がりを求める。
    中心モーメント μ20, μ02, μ11 から
      向き θ = 1/2 atan2(2μ11, μ20 - μ02)
      主軸方向の分散 λ1, λ2 = (μ20 + μ02)/2 ± sqrt(((μ20 - μ02)/2)² + μ11²)  （面積 m00 で割る）
    戻り値: (重心x, 重心y, θ[rad], 長軸の半径, 短軸の半径)  半径は 2√λ（一様な楕円なら輪郭の大きさ）"""
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    m00 = max(m["m00"], 1e-6)
    mu20, mu02, mu11 = m["mu20"] / m00, m["mu02"] / m00, m["mu11"] / m00
    theta = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    common = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    l1, l2 = (mu20 + mu02) / 2 + common, max((mu20 + mu02) / 2 - common, 0)
    return m["m10"] / m00, m["m01"] / m00, theta, 2 * np.sqrt(l1), 2 * np.sqrt(l2)


def proc_facedetect(img, st):
    # Viola-Jones 法（Haar-like 特徴 + AdaBoost + カスケード）による顔検出
    if not load_haar(st):
        return show_unavailable(img, "Haar cascade not available")
    result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 正面顔（青）と目（緑）
    faces = st.haar["face"].detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = result[y:y + h, x:x + w]
        eyes = st.haar["eye"].detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    st.found = len(faces)
    # 横顔（赤）: 学習データは片向きのみなので左右反転画像でも検出する
    if st.profile_face:
        W = gray.shape[1]
        profiles = st.haar["profile"].detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in profiles:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        flipped = cv2.flip(gray, 1)
        profiles_flipped = st.haar["profile"].detectMultiScale(flipped, 1.3, 5)
        for (x, y, w, h) in profiles_flipped:
            cv2.rectangle(result, (W - 1 - x - w, y), (W - 1 - x, y + h), (0, 0, 255), 2)
        st.found += len(profiles) + len(profiles_flipped)
    return result


def proc_bodydetect(img, st):
    if not load_haar(st):
        return show_unavailable(img, "Haar cascade not available")
    result = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bodies = st.haar["body"].detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in bodies:
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
    st.found = len(bodies)
    return result


def proc_yunet(img, st):
    # YuNet: 小さな CNN による顔検出（OpenCV 4.8 以降 / 5 の本体に含まれる）
    h, w = img.shape[:2]
    if not load_yunet(st, (w, h)):
        return show_unavailable(img, "YuNet not available")
    result = img.copy()
    _, faces = st.yunet.detect(img)
    st.found = 0 if faces is None else len(faces)
    if faces is None:
        return result
    # faces の各行: x, y, w, h, 右目x, 右目y, 左目x, 左目y, 鼻x, 鼻y, 口右x, 口右y, 口左x, 口左y, スコア
    landmark_colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 255), (0, 255, 255)]
    for f in faces:
        x, y, fw, fh = f[:4].astype(int)
        cv2.rectangle(result, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        for i in range(5):
            px, py = f[4 + 2 * i:6 + 2 * i].astype(int)
            cv2.circle(result, (px, py), 3, landmark_colors[i], -1)
        cv2.putText(result, "%.2f" % f[14], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 1)
    return result


def proc_lktracking(img, st):
    # Lucas-Kanade 法による特徴点追跡
    nxt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if st.lk_reset or st.lk_points is None or len(st.lk_points) == 0:
        st.prvs = nxt
        st.track_list = []
        st.lk_points = cv2.goodFeaturesToTrack(nxt, mask=None, **st.feature_params)
        st.lk_reset = False
        if st.lk_points is None:       # 特徴点が見つからなかった
            return img
    p0 = st.lk_points.reshape(-1, 1, 2)

    # オプティカルフローを計算
    p1, status, err = cv2.calcOpticalFlowPyrLK(st.prvs, nxt, p0, None, **st.lk_params)

    # 追跡に成功した点だけ残す
    good_new = p1[status == 1]
    good_old = p0[status == 1]

    # 軌跡を描画
    track = np.zeros_like(img)
    result = img.copy()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        color = st.lk_color[i].tolist()
        cv2.line(track, (int(a), int(b)), (int(c), int(d)), color, 2)
        cv2.circle(result, (int(a), int(b)), 5, color, -1)

    st.track_list.append(track)
    if len(st.track_list) > 20:
        st.track_list.pop(0)
    for t in st.track_list:
        result = np.where(t != 0, t, result)

    st.lk_points = good_new
    st.prvs = nxt
    return result


def proc_dog(img, st):
    # DoG (Difference of Gaussian) フィルタ: ↑↓キーでガウシアンの大きさを変更
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = st.gaussian_scale * 2 + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    return np.clip((gray - blurred) * 3.0 + 128.0, 0, 255).astype(np.uint8)


def proc_hist_eq(img, st):
    # ヒストグラム平坦化: ヒストグラムがなるべく平らになるように濃淡を割り当て直す
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return with_histogram(cv2.equalizeHist(gray), st)


def proc_hist_eq_clahe(img, st):
    if st.clahe is None:
        st.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return with_histogram(st.clahe.apply(gray), st)


# ----------------------------------------------------------------------
# 人体の骨格推定（中身は pose_estimation.py を参照）
# ----------------------------------------------------------------------
def load_pose(st):
    """骨格推定器を読み込む。使えなければ False を返す"""
    if st.pose is None:
        import pose_estimation
        missing = [p for p in (pose_estimation.PERSON_DET_MODEL, pose_estimation.POSE_MODEL)
                   if not os.path.exists(p)]
        if missing:
            print("骨格推定のモデルがありません: %s" % ", ".join(os.path.basename(p) for p in missing))
            print("python download_models.py でダウンロードしてください。")
            st.pose = False
        else:
            st.pose = pose_estimation.PoseEstimator()
    return st.pose is not False


def proc_pose(img, st):
    # 人体の骨格推定（33 点）。k キーで時間方向の平滑化を ON/OFF
    if not load_pose(st):
        return show_unavailable(img, "Pose model not found: python download_models.py")
    import pose_estimation
    result = img.copy()
    res = st.pose.process(img)
    st.found = 0 if res is None else 1
    if res is not None:
        pose_estimation.draw_pose(result, *res)
    return result


def proc_custom(img, st):
    """
    ★ ここに自分の処理を書いてみよう（キー 'u' で実行）★
    例として、Canny エッジ検出を行っています。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges


# ----------------------------------------------------------------------
# ←/→（または , .）で変える、処理ごとのパラメータ
#   処理名: (パラメータの名前, 値の候補, 最初の値)   → で候補の次の値（効果が強い・厳しい方）へ
#   ※ 二値化（b）のしきい値だけは、大津の値を起点に手動にする別の扱い（shift_threshold）
# ----------------------------------------------------------------------
PARAMS = {
    "channels":    ("space", ["RGB", "YIQ", "HSV", "Lab"], "RGB"),
    "quantize":    ("bits", [8, 7, 6, 5, 4, 3, 2, 1], 3),
    "sampling":    ("block", [1, 2, 3, 4, 6, 8, 12, 16, 24, 32], 8),
    "labeling2":   ("min area", [10, 25, 50, 100, 200, 400, 800, 1600, 3200], 100),
    "houghline":   ("votes", [40, 60, 80, 100, 120, 150, 200, 250, 300, 400, 500, 700, 1000, 1500, 2000,
                              3000, 4000, 6000, 8000, 12000, 16000, 24000], 200),
    "houghcircle": ("votes", [20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300, 400, 600, 800, 1200, 1600,
                              2400, 3200, 4800, 6400, 9600, 12800, 19200], 100),
    "pedestrian":  ("SVM threshold", [-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5], 0.0),
}


def param_key(st, proc):
    """パラメータを覚えておく名前。Hough 変換のしきい値は、ぼかし投票の σ ごとに別々に覚える"""
    if proc in HOUGH_SIGMA_GAIN and st.hough_sigma:
        return "%s@sigma%d" % (proc, st.hough_sigma)
    return proc


def param_default_index(st, proc):
    """パラメータの最初の値の位置（Hough 変換のぼかし投票では、最初の値に σ ごとの倍率を掛けた値に近いもの）"""
    _, values, default = PARAMS[proc]
    if proc in HOUGH_SIGMA_GAIN and st.hough_sigma:
        target = default * HOUGH_SIGMA_GAIN[proc][st.hough_sigma]
        return int(np.argmin([abs(np.log(v / target)) for v in values]))
    return values.index(default)


def param(st, proc):
    """処理 proc の、いまのパラメータの値"""
    _, values, _ = PARAMS[proc]
    return values[st.param_index.get(param_key(st, proc), param_default_index(st, proc))]


def shift_param(st, proc, step):
    _, values, _ = PARAMS[proc]
    i = st.param_index.get(param_key(st, proc), param_default_index(st, proc))
    st.param_index[param_key(st, proc)] = int(np.clip(i + step, 0, len(values) - 1))


def put_outlined(img, text, org, scale, color, thickness=1):
    """黒いふち付きの文字（どんな背景でも読めるように）"""
    # ふちは同じ太さの黒い文字を少しずらして重ねて描く（太い黒い文字を下に敷く方法は、
    # OpenCV 5 では太さで文字の幅が変わるため、文字がずれてしまう）
    font = cv2.FONT_HERSHEY_SIMPLEX
    for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)):
        cv2.putText(img, text, (org[0] + dx, org[1] + dy), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def put_info(img, text, row=0):
    """画像の右上に説明の文字を描く"""
    scale = max(img.shape[0] / 600.0, 0.4)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    put_outlined(img, text, (img.shape[1] - tw - 8, int(8 + th + row * th * 1.6)), scale, (0, 255, 255))


def proc_channels(img, st):
    # 色空間のチャネル表示: 左上に元の画像、残り 3 つに各チャネルをグレースケールで並べる
    # ←/→ で色空間を切り替える（RGB → YIQ → HSV → L*a*b*）
    space = param(st, "channels")
    f = img.astype(np.float32)
    b, g, r = f[..., 0], f[..., 1], f[..., 2]
    if space == "RGB":
        chans = [(r, "R"), (g, "G"), (b, "B")]
    elif space == "YIQ":
        y = 0.299 * r + 0.587 * g + 0.114 * b
        i = 0.596 * r - 0.274 * g - 0.322 * b          # およそ -152〜152
        q = 0.211 * r - 0.523 * g + 0.312 * b          # およそ -134〜134
        chans = [(y, "Y"), (128 + i * 127 / 152, "I (gray = 0)"), (128 + q * 127 / 134, "Q (gray = 0)")]
    elif space == "HSV":
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        chans = [(hsv[..., 0] * 255 / 179, "H (0-179 -> 0-255)"), (hsv[..., 1], "S"), (hsv[..., 2], "V")]
    else:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab).astype(np.float32)   # 8 ビットでは a*, b* は 128 が 0
        chans = [(lab[..., 0], "L*"), (lab[..., 1], "a* (gray = 0)"), (lab[..., 2], "b* (gray = 0)")]
    H, W = img.shape[:2]
    h2, w2 = H // 2, W // 2
    tiles = [(img, space + "  <-/->")] + [
        (cv2.cvtColor(np.clip(c, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), name) for c, name in chans]
    result = np.zeros_like(img)
    for k, (tile, name) in enumerate(tiles):
        y, x = (k // 2) * h2, (k % 2) * w2
        cell = cv2.resize(tile, (w2, h2), interpolation=cv2.INTER_AREA)
        put_info(cell, name)
        result[y:y + h2, x:x + w2] = cell
    return result


def proc_quantize(img, st):
    # 量子化: 1 チャネルあたりのビット数を減らし、2^bits 段階の値だけで表す（←/→ でビット数）
    bits = param(st, "quantize")
    step = 1 << (8 - bits)
    result = (img // step) * step + step // 2          # 各段階は、その範囲の中央の値で表す
    put_info(result, "%d bits / channel = %d levels" % (bits, 1 << bits))
    return result


def proc_sampling(img, st):
    # 標本化: block x block 画素ごとに 1 つの値にまとめ、解像度を下げる（←/→ でブロックの大きさ）
    k = param(st, "sampling")
    H, W = img.shape[:2]
    small = cv2.resize(img, (max(W // k, 1), max(H // k, 1)), interpolation=cv2.INTER_AREA)
    result = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
    put_info(result, "block %d px -> %d x %d samples" % (k, small.shape[1], small.shape[0]))
    return result


MAX_HOUGH_LINES = 100        # 描く直線の数の上限（cv2.HoughLines は票の多い順に返す）
HOUGH_SIGMAS = [0, 1, 2, 4]   # ぼかし投票の広がり σ [セル]（w キーで順に切り替え。0 は 1 セル投票）
# ぼかし投票では周りの票も加わって山の票数が大きくなるので、σ ごとのしきい値の最初の値は、
# 1 セル投票の最初の値にこの倍率を掛けたものにする（サンプル画像で、本物の直線・円が同程度に検出される比を測った目安）。
# しきい値は σ ごとに別々に覚えるので、←/→ で調整した値は σ を切り替えても残る
HOUGH_SIGMA_GAIN = {"houghline": {0: 1, 1: 3, 2: 10, 4: 33}, "houghcircle": {0: 1, 1: 5, 2: 15, 4: 40}}


def soft_votes(acc, sigma, wrap=False):
    """ぼかし投票: 各票を、計算結果のセルに 1 票、そこから距離 d のセルに exp(-d²/2σ²) 票と、
    周りにも少ない票を入れる。1 つのセルに投票した表を、中心の重みが 1 のガウス関数でぼかすのと同じ結果。
    wrap=True（直線）では θ の両端をつなぐ（θ = 0° と 180° は、ρ の符号を反転すると同じ直線）"""
    if sigma == 0:
        return acc.astype(np.float32)
    n = int(6 * sigma) | 1
    k = cv2.getGaussianKernel(n, sigma)
    k = (k / k.max()).astype(np.float32)               # 計算結果のセル（中心）の重みを 1 にする
    a = acc.astype(np.float32)
    if wrap:
        p = n // 2
        a = np.hstack([a[::-1, -p:], a, a[::-1, :p]])
        return cv2.sepFilter2D(a, -1, k, k, borderType=cv2.BORDER_CONSTANT)[:, p:-p]
    return cv2.sepFilter2D(a, -1, k, k, borderType=cv2.BORDER_CONSTANT)


def find_peaks(acc, threshold, win_r, win_c, max_n, wrap=False):
    """投票空間から山を選ぶ（直線・円で共通）:
    1. 非極大値の抑制: 縦 ±win_r・横 ±win_c セルの範囲で票数が最大のセルだけを残す
       （wrap=True のときは θ の両端をつないで判定する）
    2. 票数がしきい値以上のものを、票の多い順に最大 max_n 個選ぶ
    戻り値: [(行, 列, 票数), ...]"""
    a = acc.astype(np.float32)
    a = a + np.arange(a.size, dtype=np.float32).reshape(a.shape) * 1e-7   # 同じ票数のセルが並ぶときの順位付け
    ext = np.hstack([a[::-1, -win_c:], a, a[::-1, :win_c]]) if wrap else a
    local_max = ext >= cv2.dilate(ext, np.ones((2 * win_r + 1, 2 * win_c + 1), np.uint8))
    if wrap:
        local_max = local_max[:, win_c:-win_c]
    rows, cols = np.nonzero(local_max & (acc >= threshold))
    order = np.argsort(-acc[rows, cols])[:max_n]
    return [(int(rows[i]), int(cols[i]), float(acc[rows[i], cols[i]])) for i in order]


def set_hough_sigma(st, sigma):
    """ぼかし投票の σ を切り替える（しきい値は σ ごとに覚えている値になる）"""
    st.hough_sigma = sigma


def voting_label(st):
    if st.hough_sigma == 0:
        return "voting: single cell"
    return "voting: blurred sigma=%d" % st.hough_sigma


HOUGH_KEYS = [("<- ->", "vote threshold"),
              ("w", "voting: single cell / blur sigma 1, 2, 4"),
              ("ESC", "reset threshold"),
              ("?", "all keys")]


def draw_key_guide(img, items, max_w, top=False):
    """画像の左下（top=True なら左上）に、この処理で使えるキーの一覧を半透明の黒い枠の中に描く（常に表示）。
    items: [(キー, 説明), ...]、max_w: 枠の幅の上限（右下のヒートマップに重ならないように）"""
    H = img.shape[0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(H / 900.0, 0.35)
    width = lambda t, sc: cv2.getTextSize(t, font, sc, 1)[0][0]
    key_w = max(width(k, scale) for k, _ in items)
    total = key_w + 12 + max(width(d, scale) for _, d in items) + 16
    if total > max_w:                                  # 入りきらなければ文字を小さくする
        scale *= max_w / total
        key_w = max(width(k, scale) for k, _ in items)
        total = max_w
    lh = int(cv2.getTextSize("Ag", font, scale, 1)[0][1] * 1.8)
    bw, bh = int(total), lh * len(items) + 8
    # 上は FPS と記録の表示の 2 行、下は「?: ヘルプ」の表示の分を空ける
    x0, y0 = 8, (int(H * 0.11) if top else H - bh - int(H * 0.06))
    roi = img[y0:y0 + bh, x0:x0 + bw]
    roi[:] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)
    for i, (k, d) in enumerate(items):
        y = y0 + 4 + lh * (i + 1) - lh // 4
        put_outlined(img, k, (x0 + 8, y), scale, (0, 255, 255))
        put_outlined(img, d, (x0 + 8 + key_w + 12, y), scale, (255, 255, 255))


def hough_accumulator(edges, max_points=15000):
    """直線の Hough 変換の投票空間を、cv2.HoughLines（ρ: 1 画素, θ: 1° 刻み）と同じ細かさで計算する。
    エッジ点 (x, y) ごとに、θ を 0〜179° で動かして ρ = x cosθ + y sinθ の位置に 1 票入れる。
    エッジ点が多いときは max_points 個に間引き、その分だけ票数を補正する（検出の票数と比べられるように）。
    戻り値: (投票数の表 [ρ + 対角線の長さ, θ], 対角線の長さ)"""
    ys, xs = np.nonzero(edges)
    ratio = 1.0
    if len(xs) > max_points:
        ratio = len(xs) / max_points
        idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs, ys = xs[idx], ys[idx]
    diag = int(np.ceil(np.hypot(*edges.shape)))
    thetas = np.deg2rad(np.arange(180))
    rho = np.round(xs[:, None] * np.cos(thetas) + ys[:, None] * np.sin(thetas)).astype(np.int64) + diag
    acc = np.bincount((rho * 180 + np.arange(180)).ravel(), minlength=(2 * diag + 1) * 180)
    return acc.reshape(2 * diag + 1, 180) * ratio, diag


def draw_hough_space(img, acc, diag, lines, votes):
    """画像の右下に投票空間のヒートマップを描く。
    水色の線: 票数がしきい値（votes）以上の範囲。この中の山が直線として検出される
    緑の点: 検出した直線の (ρ, θ)"""
    H, W = img.shape[:2]
    pw, ph = int(W * 0.42), int(H * 0.42)
    x0, y0 = W - pw - 8, H - ph - 8
    # ρ 方向を表示の高さまで縮める。平均ではなく最大値をとって、細い山が消えないようにする
    f = int(np.ceil(acc.shape[0] / ph))
    pooled = np.pad(acc, ((0, (-acc.shape[0]) % f), (0, 0))).reshape(-1, f, acc.shape[1]).max(axis=1)
    heat = np.log1p(pooled.astype(np.float32))
    heat = (heat / max(heat.max(), 1e-6) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cv2.resize(heat, (pw, ph), interpolation=cv2.INTER_LINEAR), cv2.COLORMAP_INFERNO)
    above = cv2.resize((pooled >= votes).astype(np.uint8) * 255, (pw, ph), interpolation=cv2.INTER_NEAREST)
    above = cv2.dilate(above, np.ones((3, 3), np.uint8), iterations=2)    # 1〜2 セルの鋭い山も見えるよう太らせる
    contours, _ = cv2.findContours(above, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heat, contours, -1, (255, 255, 0), 1, cv2.LINE_AA)
    for rho, theta in lines:
        px = int(theta / np.pi * (pw - 1))
        py = int((rho + diag) / (2 * diag) * (ph - 1))
        cv2.circle(heat, (px, py), 3, (0, 255, 0), -1, cv2.LINE_AA)
    img[y0:y0 + ph, x0:x0 + pw] = heat
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (255, 255, 255), 1)
    font, scale = cv2.FONT_HERSHEY_SIMPLEX, max(ph / 320.0, 0.35)
    put_outlined(img, "Hough space (votes)", (x0 + 5, y0 + 16), scale, (255, 255, 255))
    put_outlined(img, "rho", (x0 + pw - 34, y0 + ph // 2), scale, (255, 255, 255))
    put_outlined(img, "cyan: votes >= %d" % votes, (x0 + 5, y0 + 34), scale, (255, 255, 0))
    put_outlined(img, "theta 0 -> 180 deg", (x0 + 5, y0 + ph - 6), scale, (255, 255, 255))


def proc_houghline(img, st):
    # Hough 変換による直線検出（OpenCV の cv2.HoughLines）: エッジ（Canny）の点が投票し、票の多い (ρ, θ) を直線とする
    # 右下に投票空間のヒートマップ（水色の線の内側が票数のしきい値以上、緑の点が検出した直線）。
    # ←/→ で直線とみなす票数のしきい値を変えると、水色の範囲と検出される直線が変わる。
    # w で投票の仕方（1 セル投票 = cv2.HoughLines / σ = 1, 2, 4 のぼかし投票）を切り替える
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    votes = param(st, "houghline")
    sg = st.hough_sigma
    if sg == 0:
        # 1 セル投票: OpenCV の cv2.HoughLines で検出する
        found = cv2.HoughLines(edges, 1, np.pi / 180, votes)
        lines = [] if found is None else [tuple(l) for l in found[:MAX_HOUGH_LINES, 0]]
        acc, diag = hough_accumulator(edges)
        space = acc
    else:
        # ぼかし投票: 自前の投票空間をぼかし、ρ ±max(15, 3σ) 画素・θ ±max(5, 3σ) 度の範囲で
        # 票数が最大のセル（非極大値の抑制）のうち、しきい値以上のものを直線とする
        acc, diag = hough_accumulator(edges, max_points=30000)
        space = soft_votes(acc, sg, wrap=True)
        peaks = find_peaks(space, votes, max(15, 3 * sg), max(5, 3 * sg), MAX_HOUGH_LINES, wrap=True)
        lines = [(r - diag, np.deg2rad(c)) for r, c, _ in peaks]          # 行 → ρ、列 → θ（度）
    st.found = len(lines)
    result = cv2.addWeighted(img, 0.5, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.5, 0)
    L = int(np.hypot(*img.shape[:2]))
    for rho, theta in lines:
        a, b = np.cos(theta), np.sin(theta)
        x, y = a * rho, b * rho
        cv2.line(result, (int(x - L * b), int(y + L * a)), (int(x + L * b), int(y - L * a)),
                 (0, 0, 255), 2, cv2.LINE_AA)
    draw_hough_space(result, space, diag, lines, votes)
    draw_key_guide(result, HOUGH_KEYS, int(result.shape[1] * 0.58) - 24)
    put_info(result, "%d lines (votes >= %d)" % (len(lines), votes))
    put_info(result, voting_label(st), row=1)
    return result


HOUGH_CIRCLE_DP = 1.5         # 円の中心の投票の表を、画像の 1/1.5 の細かさで作る
HOUGH_CIRCLE_CANNY = 120      # 円の検出に使うエッジのしきい値（強いエッジ。弱いエッジはその半分）


def hough_circle_centers(gray, min_r, max_r, max_points=8000):
    """円の Hough 変換（勾配を使う方法、cv2.HOUGH_GRADIENT）の、中心の投票空間を同じ手順で計算する。
    エッジの各点から、勾配の向き（輪郭に垂直な向き）とその逆向きに、半径 min_r〜max_r の位置へ
    「ここが円の中心かもしれない」と 1 票ずつ入れる。円の輪郭上の点の票は、本当の中心に集まる。
    （半径は、中心が決まったあとで、エッジ点までの距離の分布から決める）
    戻り値: (投票数の表（画像の 1/dp の大きさ）, エッジ画像)"""
    dx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, borderType=cv2.BORDER_REPLICATE)
    dy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, borderType=cv2.BORDER_REPLICATE)
    edges = cv2.Canny(dx, dy, HOUGH_CIRCLE_CANNY // 2, HOUGH_CIRCLE_CANNY)
    ys, xs = np.nonzero(edges)
    vx, vy = dx[ys, xs].astype(np.float64), dy[ys, xs].astype(np.float64)
    mag = np.hypot(vx, vy)
    keep = mag >= 1.0
    xs, ys, vx, vy, mag = xs[keep], ys[keep], vx[keep], vy[keep], mag[keep]
    ratio = 1.0
    if len(xs) > max_points:                     # 点が多いときは間引き、その分だけ票数を補正する
        ratio = len(xs) / max_points
        idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs, ys, vx, vy, mag = xs[idx], ys[idx], vx[idx], vy[idx], mag[idx]
    idp = 1.0 / HOUGH_CIRCLE_DP
    H, W = gray.shape
    acols, arows = int(np.ceil(W * idp)), int(np.ceil(H * idp))
    # OpenCV と同じく 1024 倍の固定小数点で計算する（1 歩で画像上の 1 画素ずつ進む）
    x0, y0 = np.rint(xs * idp * 1024).astype(np.int64), np.rint(ys * idp * 1024).astype(np.int64)
    sx, sy = np.rint(vx * idp * 1024 / mag).astype(np.int64), np.rint(vy * idp * 1024 / mag).astype(np.int64)
    r = np.arange(min_r, max_r + 1)
    acc = np.zeros(arows * acols, np.float64)
    for sign in (1, -1):                         # 勾配の向きと、その逆向き
        ax = (x0[:, None] + sign * r * sx[:, None]) >> 10
        ay = (y0[:, None] + sign * r * sy[:, None]) >> 10
        # 表の外に出たら、その先は数えない（OpenCV と同じ）
        inside = np.cumprod((ax >= 0) & (ax < acols) & (ay >= 0) & (ay < arows), axis=1).astype(bool)
        acc += np.bincount((ay * acols + ax)[inside], minlength=arows * acols)
    return acc.reshape(arows, acols) * ratio, edges


HOUGH_CIRCLE_MIN_SUPPORT = 0.25   # 円周のうち、この割合以上にエッジ点がある円だけを残す


def estimate_radius(cx, cy, xs, ys, min_r, max_r):
    """中心 (cx, cy) からエッジ点までの距離の分布から半径を決める。
    半径ごとに「その半径の円周上にエッジ点がどれだけあるか（円周に対する割合）」を求め、最大のものを選ぶ。
    戻り値: (半径, 円周に対するエッジ点の割合)"""
    d = np.hypot(xs - cx, ys - cy)
    d = d[(d >= min_r) & (d <= max_r)]
    if len(d) == 0:
        return None, 0.0
    hist = np.bincount(np.round(d).astype(int) - min_r, minlength=max_r - min_r + 1)
    hist = np.convolve(hist, np.ones(3), mode="same")        # 輪郭の太さ・ずれを考えて 3 画素幅で数える
    radii = np.arange(min_r, max_r + 1)
    support = hist / (2 * np.pi * radii)
    best = int(np.argmax(support))
    return int(radii[best]), float(support[best])


def draw_circle_space(img, acc, centers, votes):
    """画像の右下に円の中心の投票空間のヒートマップを描く。
    水色の線: 票数がしきい値（votes）以上の範囲、緑の点: 検出した円の中心"""
    H, W = img.shape[:2]
    pw = int(W * 0.36)
    ph = int(pw * acc.shape[0] / acc.shape[1])         # 画像と同じ縦横比
    x0, y0 = W - pw - 8, H - ph - 8
    heat = np.log1p(np.maximum(acc, 0).astype(np.float32))
    heat = (heat / max(heat.max(), 1e-6) * 255).astype(np.uint8)
    heat = cv2.applyColorMap(cv2.resize(heat, (pw, ph), interpolation=cv2.INTER_LINEAR), cv2.COLORMAP_INFERNO)
    above = cv2.resize((acc >= votes).astype(np.uint8) * 255, (pw, ph), interpolation=cv2.INTER_NEAREST)
    above = cv2.dilate(above, np.ones((3, 3), np.uint8), iterations=2)    # 小さな山も見えるよう太らせる
    contours, _ = cv2.findContours(above, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(heat, contours, -1, (255, 255, 0), 1, cv2.LINE_AA)
    for x, y in centers:
        cv2.circle(heat, (int(x * pw / W), int(y * ph / H)), 3, (0, 255, 0), -1, cv2.LINE_AA)
    img[y0:y0 + ph, x0:x0 + pw] = heat
    cv2.rectangle(img, (x0, y0), (x0 + pw, y0 + ph), (255, 255, 255), 1)
    scale = max(ph / 300.0, 0.35)
    put_outlined(img, "center votes (x, y)", (x0 + 5, y0 + 16), scale, (255, 255, 255))
    put_outlined(img, "cyan: votes >= %d" % votes, (x0 + 5, y0 + 34), scale, (255, 255, 0))


def proc_houghcircle(img, st):
    # Hough 変換による円検出（勾配を使う方法）:
    # 1. エッジ（Canny）の点が、勾配の向きに沿って円の中心になりうる位置に投票する（右下のヒートマップ）
    # 2. 票の多い位置を中心とする（水色の線の内側が票数のしきい値以上、緑の点が検出した中心）
    # 3. 中心からエッジ点までの距離の分布から半径を決める
    # ←/→ で中心とみなす票数のしきい値、w で投票の仕方（1 セル / ぼかし投票）
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    H = gray.shape[0]
    votes = param(st, "houghcircle")
    min_r, max_r = max(H // 40, 5), H // 3
    acc, edges = hough_circle_centers(gray, min_r, max_r)
    space = soft_votes(acc, st.hough_sigma)
    # 非極大値の抑制の範囲: 中心どうしの最小距離（画像の高さの 1/12）と 3σ の大きい方
    win = max(int(H / 12 / HOUGH_CIRCLE_DP), 3 * st.hough_sigma)
    peaks = find_peaks(space, votes, win, win, max_n=50)
    ys, xs = np.nonzero(edges)
    circles = []
    for r, c, _ in peaks:
        cx, cy = (c + 0.5) * HOUGH_CIRCLE_DP, (r + 0.5) * HOUGH_CIRCLE_DP     # 表のセル → 画像の座標
        radius, support = estimate_radius(cx, cy, xs, ys, min_r, max_r)
        if radius is not None and support >= HOUGH_CIRCLE_MIN_SUPPORT:
            circles.append((int(round(cx)), int(round(cy)), radius))
    st.found = len(circles)
    result = cv2.addWeighted(img, 0.5, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), 0.5, 0)
    for x, y, r in circles:
        cv2.circle(result, (x, y), r, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(result, (x, y), (0, 0, 255), cv2.MARKER_CROSS, 8, 2)
    draw_circle_space(result, space, [(x, y) for x, y, _ in circles], votes)
    draw_key_guide(result, HOUGH_KEYS, int(result.shape[1] * 0.64) - 24)
    put_info(result, "%d circles (votes >= %d)" % (len(circles), votes))
    put_info(result, voting_label(st), row=1)
    return result


def load_hog(st):
    """HOG 特徴 + 学習済みの線形 SVM（歩行者検出器）を用意する。使えなければ False"""
    if st.hog is None:
        if not hasattr(cv2, "HOGDescriptor"):
            print("この OpenCV (%s) には HOGDescriptor がありません（opencv-contrib-python が必要）。" % cv2.__version__)
            st.hog = False
        else:
            st.hog = cv2.HOGDescriptor()
            st.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return st.hog is not False


HOG_CELL = 8      # HOG のセルの大きさ [画素]（OpenCV の人物検出器と同じ）
HOG_BINS = 9      # 勾配の向きの数（0〜180° を 20° ずつ。b 番目は (b + 0.5) × 20° が中心）
PEDESTRIAN_KEYS = [("<- ->", "SVM threshold"),
                   ("o", "show HOG of the whole image"),
                   ("ESC", "reset threshold"),
                   ("?", "all keys")]


def hog_glyphs(size):
    """HOG の向きごとの絵（size x size 画素）。b 番目の向きの勾配は、それに垂直なエッジの向きの線分で表す"""
    glyphs = np.zeros((HOG_BINS, size, size), np.float32)
    c = (size - 1) / 2.0
    for b in range(HOG_BINS):
        a = np.deg2rad((b + 0.5) * 180.0 / HOG_BINS)          # 勾配の向き（画像の座標で、y は下向き）
        dx, dy = -np.sin(a) * size / 2, np.cos(a) * size / 2   # エッジの向き（勾配に垂直）
        p1 = (int(round((c - dx) * 16)), int(round((c - dy) * 16)))
        p2 = (int(round((c + dx) * 16)), int(round((c + dy) * 16)))
        cv2.line(glyphs[b], p1, p2, 1.0, max(size // 8, 1), cv2.LINE_AA, shift=4)
    return glyphs


def hog_cells(desc, ncx, ncy, total=False):
    """HOGDescriptor.compute の結果を、セルごとの向きのヒストグラム [セルの行, 列, 向き] に戻す。
    特徴は 2x2 セルのブロックごとに正規化して並べてあり（ブロックもブロック内のセルも列の順）、
    1 つのセルは最大 4 つのブロックに含まれるので、それらの平均をとる（total=True なら合計）"""
    nbx, nby = ncx - 1, ncy - 1
    d = np.asarray(desc, np.float32).reshape(nbx, nby, 2, 2, HOG_BINS)   # [ブロック x, y, セル x, y, 向き]
    cells = np.zeros((ncy, ncx, HOG_BINS), np.float32)
    count = np.zeros((ncy, ncx, 1), np.float32)
    for i in (0, 1):
        for j in (0, 1):
            cells[j:j + nby, i:i + nbx] += d[:, :, i, j].transpose(1, 0, 2)
            count[j:j + nby, i:i + nbx] += 1
    return cells if total else cells / count


def render_hog(cells, size, scale=None):
    """セルごとの向きのヒストグラムを、セルごとに 9 本の線分（明るさが大きさ）の絵にする。
    scale: 明るさ 1 にする値（None なら全体の上位 0.5% の値）。戻り値: 0〜1 のグレースケール画像"""
    v = np.maximum(cells, 0)
    if scale is None:
        scale = np.percentile(v, 99.5)
    v = v / max(scale, 1e-6)
    ny, nx = cells.shape[:2]
    out = np.einsum("yxb,bij->yixj", v, hog_glyphs(size)).reshape(ny * size, nx * size)
    return np.clip(out, 0, 1)


def hog_of_image(img):
    """画像全体の、8x8 画素のセルごとの勾配の向きのヒストグラム（HOG のブロック正規化の前の段階）。
    各画素の勾配の大きさを、向きの近い 2 つの向き（20° 刻み）に比例配分して足し合わせる。
    （正規化の後の特徴は、平らな所の小さな勾配まで大きくなって全体が一様に見えるので、画像全体はこちらで表示する）"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)       # [-1, 0, 1] の差分（HOG と同じ）
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    pos = (ang % 180.0) / (180.0 / HOG_BINS) - 0.5         # b 番目の向きの中心が pos = b
    b0 = np.floor(pos)
    frac = pos - b0
    b0 = b0.astype(int) % HOG_BINS
    ncx, ncy = img.shape[1] // HOG_CELL, img.shape[0] // HOG_CELL
    mag, frac, b0 = [m[:ncy * HOG_CELL, :ncx * HOG_CELL] for m in (mag, frac, b0)]
    cells = np.zeros((ncy, ncx, HOG_BINS), np.float32)
    for b in range(HOG_BINS):
        w = mag * ((b0 == b) * (1 - frac) + (b0 == (b - 1) % HOG_BINS) * frac)
        cells[..., b] = cv2.resize(w, (ncx, ncy), interpolation=cv2.INTER_AREA)
    return cells


PERSON_TILES = 4   # 人物の窓の図の数（窓の画像・HOG・SVM の重み・寄与）


def person_panel(H, W):
    """人物の窓の図を並べる場所: (セルの大きさ, 図の幅, 図の高さ, 左端 x, 上端 y, 図の間隔)"""
    size = max(int(H * 0.5) // 16, 4)                      # 窓は 8 x 16 セル
    tw, th, gap = 8 * size, 16 * size, 6
    return size, tw, th, W - PERSON_TILES * tw - (PERSON_TILES - 1) * gap - 8, H - th - 8, gap


def draw_person_hog(result, small, rect, st):
    """検出した人物の窓について、画面の右下に 4 つの図を並べる:
    window: 窓の画像（64x128 に縮めたもの）、HOG: その HOG 特徴、
    SVM w > 0: 学習済みの SVM の重み（正の部分。検出器が探している形で、入力によらず一定）、
    HOG x w: 特徴と重みの積（寄与）。緑はプラス（人物らしさを上げた）、赤はマイナス（下げた）。
    寄与をすべて足してバイアス b を加えたものが、人物らしさの値（score = HOG . w + b）になる"""
    x, y, w, h = rect
    win = st.hog.winSize                                   # (64, 128)
    crop = cv2.resize(small[max(y, 0):y + h, max(x, 0):x + w], win, interpolation=cv2.INTER_AREA)
    feat = st.hog.compute(crop).ravel()
    svm = np.asarray(st.hog.getSVMDetector() if hasattr(st.hog, "getSVMDetector")
                     else st.hog.svmDetector, np.float32).ravel()
    contrib = feat * svm[:-1]
    bias = float(svm[-1])
    score = float(contrib.sum() + bias)
    ncx, ncy = win[0] // HOG_CELL, win[1] // HOG_CELL
    H, W = result.shape[:2]
    size, tw, th, x0, y0, gap = person_panel(H, W)
    c = hog_cells(contrib, ncx, ncy, total=True)           # セルごと・向きごとの寄与の合計
    cscale = np.percentile(np.abs(c), 99.5)
    pos, neg = render_hog(c, size, cscale), render_hog(-c, size, cscale)
    contrib_img = (np.dstack([np.zeros_like(pos), pos, neg]) * 255).astype(np.uint8)   # BGR: 緑 = +、赤 = -
    tiles = [(cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR), "window"),
             (render_hog(hog_cells(feat, ncx, ncy), size), "HOG"),
             (render_hog(hog_cells(svm[:-1], ncx, ncy), size), "SVM w > 0"),
             (contrib_img, "HOG x w")]
    fscale = max(H / 900.0, 0.3)
    for k, (tile, name) in enumerate(tiles):
        if tile.dtype != np.uint8:
            tile = cv2.cvtColor((tile * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        tx = x0 + k * (tw + gap)
        result[y0:y0 + th, tx:tx + tw] = tile
        cv2.rectangle(result, (tx, y0), (tx + tw, y0 + th), (255, 255, 255), 1)
        put_outlined(result, name, (tx + 3, y0 + th - 5), fscale, (0, 255, 255))
    lh = int(cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, fscale, 1)[0][1] * 1.8)
    tx = x0 + 3 * (tw + gap)
    put_outlined(result, "score = HOG . w + b = %.2f" % score, (x0, y0 - 8 - lh), fscale, (0, 255, 255))
    put_outlined(result, "green +%.2f" % contrib[contrib > 0].sum(), (x0, y0 - 8), fscale, (0, 255, 0))
    put_outlined(result, "red %.2f" % contrib[contrib < 0].sum(), (x0 + int(tw * 1.25), y0 - 8), fscale, (80, 80, 255))
    put_outlined(result, "b %.2f" % bias, (x0 + int(tw * 2.3), y0 - 8), fscale, (255, 255, 255))


def proc_pedestrian(img, st):
    # HOG 特徴 + 線形 SVM による人物（歩行者）検出。64x128 画素の窓を動かしながら、
    # 窓内の HOG 特徴を SVM で「人／人でない」に分類する（全身が映っている必要がある）
    # ←/→ で SVM の判定のしきい値（大きいほど厳しい）、o で画像全体の HOG 特徴の表示
    # 人物を検出したら、いちばん人物らしい窓の画像・HOG 特徴・SVM の重み・寄与（特徴 x 重み）を右下に表示する
    if not load_hog(st):
        return show_unavailable(img, "HOG + SVM not available")
    s = min(1.0, 640.0 / img.shape[1])                  # 速さのため幅 640 画素以下にして検出
    small = cv2.resize(img, None, fx=s, fy=s) if s < 1.0 else img
    thr = param(st, "pedestrian")
    rects, weights = st.hog.detectMultiScale(small, hitThreshold=thr, winStride=(8, 8),
                                             padding=(8, 8), scale=1.05)
    weights = np.ravel(weights) if len(rects) else []
    st.found = len(rects)
    if st.hog_view:
        # 画像全体の HOG: 8x8 画素のセルごとに、勾配の向きのヒストグラムを線分の明るさで表す
        g = render_hog(hog_of_image(small), HOG_CELL)
        g = cv2.resize(g, (int(g.shape[1] / s), int(g.shape[0] / s)), interpolation=cv2.INTER_LINEAR)
        result = (img * 0.25).astype(np.uint8)
        hy, hx = min(g.shape[0], img.shape[0]), min(g.shape[1], img.shape[1])
        result[:hy, :hx] = np.maximum(result[:hy, :hx], (g[:hy, :hx, None] * 255).astype(np.uint8))
    else:
        result = img.copy()
    for (x, y, w, h), wt in zip(rects, weights):
        x, y, w, h = [int(v / s) for v in (x, y, w, h)]
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(result, "%.2f" % wt, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    if len(rects):
        best = int(np.argmax(weights))
        draw_person_hog(result, small, rects[best], st)
        bx, by, bw, bh = [int(v / s) for v in rects[best]]
        cv2.rectangle(result, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
    put_info(result, "%d persons (SVM threshold %.2f)" % (len(rects), thr))
    put_info(result, "HOG view (o: off)" if st.hog_view else "o: HOG view", row=1)
    draw_key_guide(result, PEDESTRIAN_KEYS, int(result.shape[1] * 0.45), top=True)   # 右下は人物の窓の図
    return result


# ----------------------------------------------------------------------
# キーと処理の対応表   キー: (処理名, 関数, 説明)
# ----------------------------------------------------------------------
PROCS = {
    'c': ("color",         proc_color,         "カラー（そのまま）"),
    'g': ("gray",          proc_gray,          "グレースケール（輝度 Y：YIQ の Y）"),
    'x': ("sobelx",        proc_sobelx,        "Sobel フィルタ（x方向）"),
    'y': ("sobely",        proc_sobely,        "Sobel フィルタ（y方向）"),
    's': ("sobel",         proc_sobel,         "勾配強度"),
    'a': ("angle",         proc_angle,         "勾配方向（色相）と強度（明度）"),
    'h': ("hue",           proc_hue,           "色相のみ表示"),
    'S': ("saturation",    proc_saturation,    "彩度のみ表示（HSV の S）"),
    'V': ("value",         proc_value,         "明度のみ表示（HSV の V）"),
    'v': ("channels",      proc_channels,      "色空間のチャネル（RGB・YIQ・HSV・Lab）"),
    'Q': ("quantize",      proc_quantize,      "量子化（ビット数）"),
    'Z': ("sampling",      proc_sampling,      "標本化（ブロックの大きさ）"),
    'H': ("houghline",     proc_houghline,     "Hough 変換: 直線と投票空間"),
    'O': ("houghcircle",   proc_houghcircle,   "Hough 変換: 円と中心の投票空間"),
    'P': ("pedestrian",    proc_pedestrian,    "人物検出（HOG + SVM）"),
    'b': ("binary",        proc_binary,        "二値化（大津の方法）"),
    ':': ("sift",          proc_sift,          "SIFT 特徴点"),
    'f': ("denseflow",     proc_denseflow,     "密なオプティカルフロー（Farneback）"),
    'l': ("labeling",      proc_labeling,      "ラベリング"),
    ';': ("labeling2",     proc_labeling2,     "ラベリング＋面積・重心・主軸"),
    't': ("lktracking",    proc_lktracking,    "特徴点追跡（Lucas-Kanade）"),
    'F': ("facedetect",    proc_facedetect,    "顔検出（Haar-like 特徴）"),
    'Y': ("yunet",         proc_yunet,         "顔検出（YuNet・CNN）"),
    'K': ("pose",          proc_pose,          "人体の骨格推定（MediaPipe Pose）"),
    'U': ("bodydetect",    proc_bodydetect,    "上半身検出（Haar-like）"),
    'd': ("dog",           proc_dog,           "DoG フィルタ"),
    'e': ("hist_eq",       proc_hist_eq,       "ヒストグラム平坦化"),
    'E': ("hist_eq_CLAHE", proc_hist_eq_clahe, "適応的ヒストグラム平坦化（CLAHE）"),
    'u': ("custom",        proc_custom,        "自作の処理（proc_custom を編集）"),
}
PROC_FUNCS = {name: func for (name, func, _) in PROCS.values()}

# ヘルプでの処理の分類: (見出し, 英語の見出し, キー)
PROC_GROUPS = [
    ("色・濃淡", "Color & intensity", "cgvhSVQZ"),
    ("ヒストグラム・領域", "Histogram & regions", "beEl;"),
    ("特徴・フィルタ", "Features & filters", "xysad:HO"),
    ("検出", "Detection", "FYUPK"),
    ("動き", "Motion", "ft"),
    ("自作", "Your own", "u"),
]
assert sorted("".join(g for _, _, g in PROC_GROUPS)) == sorted(PROCS), "PROC_GROUPS と PROCS が合っていない"
NEAREST_PROCS = {"quantize", "sampling"}     # 表示の拡大で画素の境目をぼかさない処理

BLURS = {
    'G': ("gaussian",  "前処理: ガウシアンぼかし"),
    'm': ("median",    "前処理: メディアンフィルタ"),
    'B': ("bilateral", "前処理: バイラテラルフィルタ"),
}


# ----------------------------------------------------------------------
# 演習記録の項目（receipt.py）: 試せる項目の一覧（カタログ）と、今のフレームで使っている項目
#   proc:処理名 / param:処理名=値（←/→）/ set:処理名/設定=値（表示の切り替え）/ blur:種類（前処理）
#   機能を追加したら、ここに項目を加えると網羅の割合に反映される（古い記録もそのまま読める）
# ----------------------------------------------------------------------
HIST_PROCS = ("gray", "binary", "hist_eq", "hist_eq_CLAHE")   # i でヒストグラムを表示できる処理


def record_catalog():
    items = ["proc:" + name for name, _, _ in PROCS.values()]
    for proc, (_, values, _) in PARAMS.items():
        items += ["param:%s=%s" % (proc, v) for v in values]
    items += ["param:binary=otsu", "param:binary=manual"]
    for proc in ("houghline", "houghcircle"):
        items += ["set:%s/sigma=%d" % (proc, sg) for sg in HOUGH_SIGMAS]
    for proc in HIST_PROCS:
        items += ["set:%s/hist=on" % proc, "set:%s/hist=off" % proc]
    items += ["set:pedestrian/hog=on", "set:pedestrian/hog=off",
              "set:facedetect/profile=on", "set:facedetect/profile=off",
              "set:pose/smoothing=on", "set:pose/smoothing=off"]
    items += ["blur:" + name for name, _ in BLURS.values()]
    return items


def record_items(st):
    """今のフレームで使っている項目（record_catalog の名前）"""
    proc = st.proc
    items = ["proc:" + proc]
    if proc in PARAMS:
        items.append("param:%s=%s" % (proc, param(st, proc)))
    if proc == "binary":
        items.append("param:binary=" + ("otsu" if st.bin_manual is None else "manual"))
    if proc in ("houghline", "houghcircle"):
        items.append("set:%s/sigma=%d" % (proc, st.hough_sigma))
    if proc in HIST_PROCS:
        items.append("set:%s/hist=%s" % (proc, "on" if st.show_hist else "off"))
    if proc == "pedestrian":
        items.append("set:pedestrian/hog=" + ("on" if st.hog_view else "off"))
    if proc == "facedetect":
        items.append("set:facedetect/profile=" + ("on" if st.profile_face else "off"))
    if proc == "pose" and st.pose:
        items.append("set:pose/smoothing=" + ("on" if st.pose.smoothing else "off"))
    strength = {"gaussian": st.gaussian_scale, "median": st.median_scale, "bilateral": st.bilateral_scale}
    if strength.get(st.blur_proc, 1) > 1:                # ぼかしが実際にかかっているとき
        items.append("blur:" + st.blur_proc)
    return items


def print_help():
    print("-" * 60)
    print("キー操作（画像ウィンドウを選択した状態で押してください）")
    for title, _, keys in PROC_GROUPS:
        print("【%s】" % title)
        for k in keys:
            name = PROCS[k][0]
            extra = ("  （←/→: %s）" % PARAMS[name][0]) if name in PARAMS else (
                "  （←/→: しきい値を手動に）" if name == "binary" else "")
            print("  %s : %s%s" % (k, PROCS[k][2], extra))
    print("【前処理（ぼかし）】")
    for k, (_, desc) in BLURS.items():
        print("  %s : %s" % (k, desc))
    print("  ↑ / + : ぼかしを強く        ↓ / - : ぼかしを弱く")
    print("【その他】")
    print("  ← / , → / . : 今の処理のパラメータを変える（v Q Z ; H O P）")
    print("                b ではしきい値を下げる・上げる（押すと手動になり値が固定される）")
    print("  ESC   : ぼかしと今の処理のパラメータを元に戻す（b はしきい値を自動（大津）に）")
    print("  i     : ヒストグラム表示の ON/OFF（g b e E）")
    print("  w     : Hough 変換の投票の広がりを切り替え（H O。1 セル → σ=1 → 2 → 4 セルのぼかし投票）")
    print("  p     : 顔検出中に横顔検出の ON/OFF")
    print("  k     : 骨格推定中に平滑化の ON/OFF")
    print("  n     : NDI 送信の ON/OFF（ndi-python が必要・任意）")
    print("  R     : 演習記録をすぐ保存（自動でも保存）")
    print("  ?     : ヘルプ表示の ON/OFF（ウィンドウ）")
    print("  q     : 終了")
    print("-" * 60)


# ----------------------------------------------------------------------
# ウィンドウへのヘルプ表示（? キーで ON/OFF）
#   OpenCV 5 以降は内蔵フォントで日本語を描けるので日本語で、
#   OpenCV 4 では日本語が描けない（? になる）ので英語で表示する。
# ----------------------------------------------------------------------
HELP_EN = {   # OpenCV 4 用の英語の説明（ない処理は処理名をそのまま表示）
    "color": "Color (as is)", "gray": "Grayscale (luma Y of YIQ)",
    "sobelx": "Sobel filter (x)", "sobely": "Sobel filter (y)",
    "sobel": "Gradient magnitude", "angle": "Gradient direction (hue) & magnitude",
    "hue": "Hue only", "saturation": "Saturation only (S of HSV)", "value": "Value only (V of HSV)",
    "channels": "Color channels (RGB YIQ HSV Lab)", "quantize": "Quantization (bits)",
    "sampling": "Sampling (block size)",
    "houghline": "Hough transform: lines & vote space", "houghcircle": "Hough transform: circles & center votes",
    "pedestrian": "Person detection (HOG + SVM)",
    "binary": "Binarize (Otsu)", "sift": "SIFT keypoints",
    "denseflow": "Dense optical flow (Farneback)", "labeling": "Labeling",
    "labeling2": "Labeling + area, centroid, principal axis",
    "lktracking": "Feature tracking (Lucas-Kanade)",
    "facedetect": "Face detection (Haar-like)",
    "yunet": "Face detection (YuNet, CNN)", "bodydetect": "Upper body detection (Haar-like)",
    "pose": "Human pose estimation (MediaPipe Pose)",
    "dog": "DoG filter", "hist_eq": "Histogram equalization",
    "hist_eq_CLAHE": "Adaptive histogram equalization (CLAHE)",
    "custom": "Your own processing (edit proc_custom)",
    "gaussian": "Gaussian blur", "median": "Median filter", "bilateral": "Bilateral filter",
}

COLOR_HEAD = (255, 255, 0)      # 見出し（水色）
COLOR_ITEM = (255, 255, 255)    # 通常（白）
COLOR_CURRENT = (0, 255, 255)   # 選択中の処理（黄色）
COLOR_UNTRIED = (0, 150, 255)   # まだ試していない機能（橙。演習記録をとっているとき）


class TextDrawer:
    """OpenCV 4 / 5 の違いを吸収して文字を描くクラス（px は文字の大きさ[画素]）"""

    # 日本語を描くためのフォント候補（先に見つかったものを使う）
    # OpenCV 5 の FontFace はフォント名（"sans" など）を解決せず、英字だけの内蔵フォントに
    # なってしまう（日本語は 1 文字 1 個の "?" になる）。ファイルのパスを渡したときだけ
    # 実際のフォントが使われるので、ここではパスを直接指定する。
    JP_FONT_PATHS = (
        r"C:\Windows\Fonts\YuGothM.ttc",       # Windows: Yu Gothic
        r"C:\Windows\Fonts\meiryo.ttc",        # Windows: メイリオ
        r"C:\Windows\Fonts\msgothic.ttc",      # Windows: MS ゴシック
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",   # macOS
        "/System/Library/Fonts/Hiragino Sans GB.ttc",        # macOS
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Linux
    )

    def __init__(self):
        self.face = None
        self.japanese = False
        if hasattr(cv2, "FontFace"):        # OpenCV 5 以降
            for path in self.JP_FONT_PATHS:     # 日本語が描けるフォントを探す
                if not os.path.exists(path):
                    continue
                try:
                    self.face = cv2.FontFace(path)
                    self.japanese = True
                    break
                except cv2.error:
                    continue
            if self.face is None:               # 無ければ英字だけの内蔵フォント
                try:
                    self.face = cv2.FontFace("sans")
                except cv2.error:
                    self.face = None

    def _hershey_scale(self, px):
        return cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, max(int(px * 0.7), 1), 1)

    def width(self, text, px):
        if self.face is not None:
            return cv2.getTextSize((0, 0), text, (0, 0), self.face, int(px))[2]
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self._hershey_scale(px), 1)
        return w

    def put(self, img, text, org, color, px, bold=False):
        org = (int(org[0]), int(org[1]))
        if self.face is not None:
            cv2.putText(img, text, org, color, self.face, int(px), 700 if bold else 400)
        else:
            cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, self._hershey_scale(px),
                        color, 2 if bold else 1, cv2.LINE_AA)


def help_columns(st, ja):
    """ヘルプの内容を (キー, 説明, 色) のリストの列（3 列）として返す"""
    def desc_of(name, desc):
        return desc if ja else HELP_EN.get(name, name)

    def group(i):
        title, title_en, keys = PROC_GROUPS[i]
        rows = [("", title if ja else title_en, COLOR_HEAD)]
        for k in keys:
            name, _, desc = PROCS[k]
            mark = (" ←→" if ja else " <->") if (name in PARAMS or name == "binary") else ""
            prog = st.progress.get("param:" + name) if st.tried is not None else None
            if prog:                                         # ←/→ の値のうち試した数 / 必要な数
                mark += " %d/%d" % (min(prog[0], prog[1]), prog[1])
            color = COLOR_ITEM
            if name == st.proc:
                color = COLOR_CURRENT
            elif st.tried is not None and "proc:" + name not in st.tried:
                color = COLOR_UNTRIED
            rows.append((k, desc_of(name, desc) + mark, color))
        return rows

    blank = [("", "", COLOR_ITEM)]
    col1 = group(0) + blank + group(1)
    col2 = group(2) + blank + group(3) + blank + group(4) + blank + group(5)

    col3 = [("", "前処理（ぼかし）" if ja else "Pre-filter (blur)", COLOR_HEAD)]
    for k, (name, desc) in BLURS.items():
        desc = desc.replace("前処理: ", "")
        col3.append((k, desc_of(name, desc), COLOR_CURRENT if name == st.blur_proc else COLOR_ITEM))
    blur_param = {"gaussian": "k=%d" % st.gaussian_scale,
                  "median": "k=%d" % st.median_scale,
                  "bilateral": "d=%d" % st.bilateral_scale}.get(st.blur_proc, "")
    profile = "ON" if st.profile_face else "OFF"
    smooth = "OFF" if (st.pose and not st.pose.smoothing) else "ON"
    hist = "ON" if st.show_hist else "OFF"
    hogv = "ON" if st.hog_view else "OFF"
    soft = ("σ=%d" if ja else "sigma=%d") % st.hough_sigma if st.hough_sigma else "OFF"
    if ja:
        col3 += [("↑ / +", "ぼかしを強く", COLOR_ITEM),
                 ("↓ / -", "ぼかしを弱く", COLOR_ITEM),
                 ("", "（現在: %s  %s）" % (st.blur_proc, blur_param), COLOR_CURRENT),
                 ("", "", COLOR_ITEM),
                 ("", "その他", COLOR_HEAD),
                 ("← →", "←→ の付いた処理のパラメータ", COLOR_ITEM),
                 ("", "を変える（, . でも可）", COLOR_ITEM),
                 ("ESC", "ぼかし・パラメータを元に戻す", COLOR_ITEM),
                 ("i", "ヒストグラム %s（g b e E）" % hist, COLOR_ITEM),
                 ("w", "Hough のぼかし投票 %s（H O）" % soft, COLOR_ITEM),
                 ("o", "HOG 特徴の表示 %s（P の実行中）" % hogv, COLOR_ITEM),
                 ("p", "横顔検出 %s（F の実行中）" % profile, COLOR_ITEM),
                 ("k", "骨格の平滑化 %s（K の実行中）" % smooth, COLOR_ITEM),
                 ("n", "NDI 送信 ON/OFF（任意機能）", COLOR_ITEM),
                 ("R", "演習記録をすぐ保存（自動でも保存）", COLOR_ITEM),
                 ("?", "このヘルプを閉じる / 開く", COLOR_ITEM),
                 ("q", "終了", COLOR_ITEM)]
    else:
        col3 += [("Up / +", "stronger blur", COLOR_ITEM),
                 ("Down / -", "weaker blur", COLOR_ITEM),
                 ("", "(now: %s  %s)" % (st.blur_proc, blur_param), COLOR_CURRENT),
                 ("", "", COLOR_ITEM),
                 ("", "Others", COLOR_HEAD),
                 ("<- ->", "change parameter of <-> items", COLOR_ITEM),
                 ("", "(also , .)", COLOR_ITEM),
                 ("ESC", "reset blur & parameter", COLOR_ITEM),
                 ("i", "histogram %s (g b e E)" % hist, COLOR_ITEM),
                 ("w", "Hough soft voting %s (H O)" % soft, COLOR_ITEM),
                 ("o", "HOG view %s (during P)" % hogv, COLOR_ITEM),
                 ("p", "profile face %s (during F)" % profile, COLOR_ITEM),
                 ("k", "pose smoothing %s (during K)" % smooth, COLOR_ITEM),
                 ("n", "NDI output ON/OFF (optional)", COLOR_ITEM),
                 ("R", "save the practice record now", COLOR_ITEM),
                 ("?", "close / open this help", COLOR_ITEM),
                 ("q", "quit", COLOR_ITEM)]
    if st.tried is not None:
        col3 += [("", "", COLOR_ITEM),
                 ("", "橙: まだ試していない機能" if ja else "orange: not tried yet", COLOR_UNTRIED),
                 ("", "（n/m: ←→ で試した値の数 / 目標）" if ja else "(n/m: values tried with <-> / goal)", COLOR_UNTRIED)]
    return [col1, col2, col3]


def status_px(H):
    """画面上の FPS 表示などの文字サイズ（画像の高さに合わせる）"""
    return max(H / 30, 12)


def render_help_layer(W, H, st, drawer):
    """黒地にヘルプの文字を描いた画像を作る"""
    ja = drawer.japanese
    title = ("キー操作（ウィンドウを選択した状態で押す）" if ja
             else "Key operations (click the window first)")
    cols = help_columns(st, ja)
    rows = max(len(c) for c in cols) + 2          # +2 はタイトルと空行の分

    def col_size(col, px):
        """(キー欄の幅, 列全体の幅) を返す。見出しはキー欄を使わない"""
        kw = max(drawer.width(k, px) for k, _, _ in col if k) + px
        w = max((kw if k else 0) + drawer.width(t, px) for k, t, _ in col if t)
        return kw, max(w, max(drawer.width(t, px) for k, t, _ in col if not k and t))

    # 上端は FPS 表示の行のために空けておき、残りの高さに収まる文字サイズを決める。
    # 横にはみ出す場合はさらに縮める
    top = status_px(H) * 1.8
    area_h = H - top - H * 0.03
    px = min(max(area_h / rows / 1.45, 9), 40)
    for _ in range(4):
        sizes = [col_size(c, px) for c in cols]
        gap = px * 2.0
        total_w = max(sum(w for _, w in sizes) + gap * (len(cols) - 1), drawer.width(title, px * 1.1))
        if total_w <= W * 0.94:
            break
        px *= W * 0.94 / total_w
    line_h = px * 1.45

    layer = np.zeros((H, W, 3), np.uint8)
    x = (W - total_w) / 2
    y0 = top + (area_h - rows * line_h) / 2 + px
    drawer.put(layer, title, (x, y0), COLOR_ITEM, px * 1.1, bold=True)
    for col, (kw, w) in zip(cols, sizes):
        y = y0 + line_h * 2
        for k, text, color in col:
            bold = (color == COLOR_HEAD)
            if k:
                drawer.put(layer, k, (x, y), color, px, bold)
                drawer.put(layer, text, (x + kw, y), color, px, bold)
            elif text:
                drawer.put(layer, text, (x, y), color, px, bold)
            y += line_h
        x += w + gap
    return layer


def draw_help(img, st, drawer, cache):
    """画像を暗くしてヘルプを重ねる（文字の画像は内容が変わったときだけ作り直す）"""
    H, W = img.shape[:2]
    key = (W, H, st.proc, st.blur_proc, st.gaussian_scale, st.median_scale,
           st.bilateral_scale, st.profile_face, bool(st.pose and st.pose.smoothing), st.show_hist,
           st.hough_sigma, st.hog_view, len(st.tried) if st.tried is not None else -1, sum(a for a, _ in st.progress.values()))
    if cache.get("key") != key:
        cache["key"] = key
        cache["layer"] = render_help_layer(W, H, st, drawer)
    dark = cv2.convertScaleAbs(img, alpha=HELP_DIM)
    return cv2.max(dark, cache["layer"])


# ----------------------------------------------------------------------
# キー入力（OS によって矢印キーのコードが異なるので吸収する）
# ----------------------------------------------------------------------
KEY_UP_CODES = {2490368}      # Windows
KEY_DOWN_CODES = {2621440}    # Windows
KEY_UP_LOW16 = {0xF700, 0xFF52}    # macOS, Linux
KEY_DOWN_LOW16 = {0xF701, 0xFF54}  # macOS, Linux
KEY_LEFT_CODES, KEY_RIGHT_CODES = {2424832}, {2555904}        # Windows
KEY_LEFT_LOW16, KEY_RIGHT_LOW16 = {0xF702, 0xFF51}, {0xF703, 0xFF53}   # macOS, Linux


def read_key():
    """溜まっているキー入力のうち最新のものだけを返す（それ以外は捨てる）"""
    code = -1
    while True:
        k = cv2.waitKeyEx(1)
        if k == -1:
            break
        code = k
    if code == -1:
        return None
    low16 = code & 0xFFFF
    if code in KEY_UP_CODES or low16 in KEY_UP_LOW16:
        return "UP"
    if code in KEY_DOWN_CODES or low16 in KEY_DOWN_LOW16:
        return "DOWN"
    if code in KEY_LEFT_CODES or low16 in KEY_LEFT_LOW16:
        return "LEFT"
    if code in KEY_RIGHT_CODES or low16 in KEY_RIGHT_LOW16:
        return "RIGHT"
    if 0xF700 <= low16 <= 0xF8FF or 0xFF00 <= low16 <= 0xFFFF:
        return None  # その他の特殊キーは無視
    return chr(code & 0xFF)


# ----------------------------------------------------------------------
# NDI 送信（任意機能。ndi-python がインストールされている場合のみ）
# ----------------------------------------------------------------------
ndi = None


def NDI_setup():
    global ndi
    try:
        import NDIlib
        ndi = NDIlib
    except ModuleNotFoundError as e:
        print(e)
        print("NDI module not found, so this function is unavailable.")
        return None, None
    if not ndi.initialize():
        print("NDI module found but cannot be initialized.")
        return None, None
    print("NDI successfully initialized.")
    send_settings = ndi.SendCreate()
    send_settings.ndi_name = "ndi-python"
    ndi_send = ndi.send_create(send_settings)
    ndi_frame = ndi.VideoFrameV2()
    return ndi_send, ndi_frame


def NDI_finish(ndi_send):
    if ndi_send is not None:
        ndi.send_destroy(ndi_send)
        ndi.destroy()


# ----------------------------------------------------------------------
# メイン
# ----------------------------------------------------------------------
def open_source(arg):
    """引数が数字ならカメラ番号、そうでなければ動画ファイルとして開く"""
    if arg is None:
        arg = "0"
    if arg.isdigit():
        print("camera device num: %s" % arg)
        cap = cv2.VideoCapture(int(arg))
        cap.set(cv2.CAP_PROP_FPS, 30)
        return cap, False
    print("video file: %s" % arg)
    return cv2.VideoCapture(arg), True


def draw_recording_status(img, recorder, drawer, cov):
    """演習記録をとっていることと、網羅の状況（これまでの実行も含む）を FPS の表示の下に出す"""
    H, W = img.shape[:2]
    px = status_px(H) * 0.7
    procs = cov.get("proc", [0, 0])
    others = [sum(cov.get(g, [0, 0])[i] for g in cov if g != "proc") for i in (0, 1)]
    total = receipt_module().format_duration(recorder.total_seconds())
    if drawer.japanese:
        text = "● 記録中 %s   機能 %d/%d   設定 %d/%d   合計 %s   （? で未体験の機能を確認）" % (
            recorder.student_id, procs[0], procs[1], others[0], others[1], total)
    else:
        text = "REC %s   features %d/%d   settings %d/%d   total %s   (? to see untried)" % (
            recorder.student_id, procs[0], procs[1], others[0], others[1], total)
    y = status_px(H) * 1.3 + px * 1.5
    width = min(int(drawer.width(text, px) + px * 1.2), W)
    y0, y1 = int(y - px * 1.15), int(y + px * 0.45)
    img[y0:y1, :width] = (img[y0:y1, :width] * 0.4).astype(np.uint8)
    drawer.put(img, text, (px * 0.7, y), (120, 160, 255), px)


def receipt_module():
    import receipt
    return receipt


def save_receipt(recorder, quiet=False):
    path, code = recorder.save()
    if path is None or quiet:
        return
    R = receipt_module()
    print("-" * 60)
    print("演習記録を保存しました: %s" % path)
    print("確認コード: %s" % code)
    cov = recorder.coverage()
    print("これまでに試した項目: " + "  ".join("%s %d/%d" % (R.GROUP_NAMES.get(g, g), a, b)
                                         for g, (a, b) in R.sort_groups(cov)))
    print("提出の指示があったら、このファイルを提出してください。")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="カメラキャプチャ＆画像処理サンプル")
    parser.add_argument("source", nargs="?", default=None, help="カメラ番号 または 動画ファイル (default: 0)")
    parser.add_argument("--id", default=None, help="学籍番号（初回に入力したものを変えるとき）")
    parser.add_argument("--task", default="", help="課題コード（指示があったとき。演習記録に付ける）")
    parser.add_argument("--no-record", action="store_true", help="演習記録をとらない")
    args = parser.parse_args()

    # 演習記録（receipt.py）: 学籍番号は初回だけ聞き、このフォルダに保存しておく
    R = receipt_module()
    recorder = None
    student_id = None
    if not args.no_record:
        R.ensure_install_id()
        student_id = R.get_student_id(args.id)

    cap, is_file = open_source(args.source)

    ret, img = cap.read()
    if not ret:
        print("カメラ（または動画）から画像を取得できませんでした。")
        print("  - カメラ番号を変えて試してください（例: python capture.py 1）")
        print("  - macOS ではターミナルにカメラへのアクセス許可が必要です")
        print("    （システム設定 > プライバシーとセキュリティ > カメラ）")
        sys.exit(1)

    print("FPS:%f" % cap.get(cv2.CAP_PROP_FPS))
    print("Image Height:%d" % int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Image Width:%d" % int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))

    height, width = img.shape[:2]
    proc_size = (int(width * PROC_SCALE), int(height * PROC_SCALE))
    st = State((proc_size[1], proc_size[0]))

    ndi_send, ndi_frame = None, None
    print_help()
    show_help = SHOW_HELP_AT_START
    drawer = TextDrawer()
    help_cache = {}
    if student_id:
        recorder = R.Recorder("capture", student_id, record_catalog(), args.task,
                              "file" if is_file else "camera", (width, height))
        print("学籍番号 %s で演習記録をとります（%s）。違う場合は --id で指定してください。"
              % (student_id, os.path.relpath(R.record_path(student_id))))
        if recorder.load_state == "broken":
            print("前の記録ファイルが壊れていたので、別の名前で残して新しく記録します。")
        st.tried, st.progress = recorder.tried(), recorder.progress()
        atexit.register(recorder.save)          # Ctrl+C などで止めたときも保存する
    coverage = recorder.coverage() if recorder else {}
    t_tried = time.perf_counter()

    t_prev = time.perf_counter()
    while True:
        t_now = time.perf_counter()
        duration = max(t_now - t_prev, 1e-6)
        t_prev = t_now

        # 画像の取り込み
        ret, img = cap.read()
        if not ret:
            if is_file:      # 動画ファイルは先頭に戻して繰り返す
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("image not captured. stop.")
            break

        # 処理速度のため縮小
        img = cv2.resize(img, proc_size)

        # キー入力（演習記録をとっているときは、どの処理の実行中に押したかを記録）
        key = read_key()
        if recorder is not None:
            recorder.on_key(key, st.proc)
        if key == 'q':
            break
        elif key == "UP" or key == '+':
            st.increase_param()
        elif key == "DOWN" or key == '-':
            st.decrease_param()
        elif key in ("LEFT", ',', "RIGHT", '.'):     # 今の処理のパラメータを変える
            step = 1 if key in ("RIGHT", '.') else -1
            if st.proc == "binary":
                shift_threshold(st, step * BIN_STEP)
            elif st.proc in PARAMS:
                shift_param(st, st.proc, step)
        elif key == chr(27):          # ESC: ぼかしと今の処理のパラメータを元に戻す
            st.reset_param()
            st.bin_manual = None
            st.param_index.pop(param_key(st, st.proc), None)
        elif key == 'i':
            st.show_hist = not st.show_hist
        elif key == 'w':
            # ぼかし投票の広がりを 0（1 セル投票）→ 1 → 2 → 4 → 0 … と切り替える
            i = HOUGH_SIGMAS.index(st.hough_sigma)
            set_hough_sigma(st, HOUGH_SIGMAS[(i + 1) % len(HOUGH_SIGMAS)])
        elif key == '?':
            show_help = not show_help
            if show_help:
                print_help()
        elif key in BLURS:
            st.blur_proc = BLURS[key][0]
            st.reset_param()
            print("blur:", st.blur_proc)
        elif key in PROCS:
            st.proc = PROCS[key][0]
            if st.proc == "lktracking":
                st.lk_reset = True
            if st.proc == "binary":
                st.bin_manual = None
            if st.proc == "pose" and st.pose:
                st.pose.reset()
            if st.proc in ("dog", "hist_eq", "hist_eq_CLAHE"):
                st.blur_proc = "gaussian"
            print("proc:", st.proc)
        elif key == 'k' and st.proc == "pose" and st.pose:
            st.pose.smoothing = not st.pose.smoothing
            st.pose.reset()
            print("pose smoothing:", "ON" if st.pose.smoothing else "OFF")
        elif key == 'o' and st.proc == "pedestrian":
            st.hog_view = not st.hog_view
            print("HOG view:", "ON" if st.hog_view else "OFF")
        elif key == 'p' and st.proc == "facedetect":
            st.profile_face = not st.profile_face
            print("profile face mode" if st.profile_face else "frontal face mode")
        elif key == 'R' and recorder is not None:
            save_receipt(recorder)
        elif key == 'n':
            if ndi_send is None:
                ndi_send, ndi_frame = NDI_setup()
            else:
                NDI_finish(ndi_send)
                ndi_send = None

        # 前処理（ぼかし）→ 画像処理
        raw = img
        img = blur(img, st)
        st.found = None
        result = PROC_FUNCS[st.proc](img, st)
        if recorder is not None:
            recorder.record(st.proc, raw, st, param(st, st.proc) if st.proc in PARAMS else None,
                            items=record_items(st))
            if t_now - t_tried >= 1.0:          # 網羅の状況は 1 秒ごとに計算し直す
                t_tried = t_now
                st.tried, st.progress = recorder.tried(), recorder.progress()
                coverage = recorder.coverage()

        # 表示
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        interp = cv2.INTER_NEAREST if st.proc in NEAREST_PROCS else cv2.INTER_LINEAR
        result = cv2.resize(result, (width, height), interpolation=interp)
        if show_help:
            result = draw_help(result, st, drawer, help_cache)
        px = status_px(height)
        status = "FPS:%.1f  proc:%s  blur:%s" % (1.0 / duration, st.proc, st.blur_proc)
        if st.proc in PARAMS:
            status += "  %s:%s" % (PARAMS[st.proc][0], param(st, st.proc))
        drawer.put(result, status, (px * 0.5, px * 1.3), (0, 255, 0), px, bold=True)
        if not show_help:
            drawer.put(result, "?: ヘルプ" if drawer.japanese else "?: help",
                       (px * 0.5, height - px * 0.6), (0, 255, 0), px * 0.8)
            if recorder is not None:
                draw_recording_status(result, recorder, drawer, coverage)
        cv2.imshow(WINDOW_NAME, result)

        # NDI へ送信
        if ndi_send is not None:
            ndi_frame.data = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
            ndi_frame.FourCC = ndi.FOURCC_VIDEO_TYPE_BGRX
            ndi.send_send_video_v2(ndi_send, ndi_frame)

    if recorder is not None:
        save_receipt(recorder)
    NDI_finish(ndi_send)
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    main()
