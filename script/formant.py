# coding:utf-8
from __future__ import division
import wave
import numpy as np
import scipy.io.wavfile
import scipy.signal
import pylab as P
from levinson_durbin import autocorr, LevinsonDurbin


"""LPCスペクトル包絡を求める"""

def wavread(filename):
    wf = wave.open(filename, "r")
    fs = wf.getframerate()
    x = wf.readframes(wf.getnframes())
    x = np.frombuffer(x, dtype="int16") / 32768.0  # (-1, 1)に正規化
    wf.close()
    return x, float(fs)

def preEmphasis(signal, p):
    """プリエンファシスフィルタ"""
    # 係数 (1.0, -p) のFIRフィルタを作成
    return scipy.signal.lfilter([1.0, -p], 1, signal)

def plot_signal(s, a, e, fs, lpcOrder, file):
    t = np.arange(0.0, len(s) / fs, 1/fs)
    # LPCで前向き予測した信号を求める
    predicted = np.copy(s)
    # 過去lpcOrder分から予測するので開始インデックスはlpcOrderから
    # それより前は予測せずにオリジナルの信号をコピーしている
    for i in range(lpcOrder, len(predicted)):
        predicted[i] = 0.0
        for j in range(1, lpcOrder):
            predicted[i] -= a[j] * s[i - j]
    # オリジナルの信号をプロット
    P.plot(t, s)
    # LPCで前向き予測した信号をプロット
    P.plot(t, predicted,"r",alpha=0.4)
    P.xlabel("Time (s)")
    P.xlim((-0.001, t[-1]+0.001))
    P.title(file)
    P.grid()
    P.show()
    return 0

def plot_spectrum(s, a, e, fs, file):
    # LPC係数の振幅スペクトルを求める
    nfft = 2048   # FFTのサンプル数
    fscale = np.fft.fftfreq(nfft, d = 1.0 / fs)[:nfft/2]
    # オリジナル信号の対数スペクトル
    spec = np.abs(np.fft.fft(s, nfft))
    logspec = 20 * np.log10(spec)
    P.plot(fscale, logspec[:nfft/2])
    # LPC対数スペクトル
    w, h = scipy.signal.freqz(np.sqrt(e), a, nfft, "whole")
    lpcspec = np.abs(h)
    loglpcspec = 20 * np.log10(lpcspec)
    P.plot(fscale, loglpcspec[:nfft/2], "r", linewidth=2)
    P.xlabel("Frequency (Hz)")
    P.xlim((-100, 8100))
    P.title(file)
    P.grid()
    P.show()
    return 0

def lpc_spectral_envelope(file):
    # 音声をロード
    wav, fs = wavread(file)
    t = np.arange(0.0, len(wav) / fs, 1/fs)
    # 音声波形の中心部分を切り出す
    center = len(wav) / 2  # 中心のサンプル番号
    cuttime = 0.04         # 切り出す長さ [s]

    s = wav[int(center - cuttime/2*fs) : int(center + cuttime/2*fs)]
    # プリエンファシスフィルタをかける
    p = 0.97         # プリエンファシス係数
    s = preEmphasis(s, p)
    # ハミング窓をかける
    hammingWindow = np.hamming(len(s))
    s = s * hammingWindow
    # LPC係数を求める
    #    lpcOrder = 12
    lpcOrder = 32
    r = autocorr(s, lpcOrder + 1)
    a, e  = LevinsonDurbin(r, lpcOrder)
    plot_signal(s, a, e, fs, lpcOrder, file)
    plot_spectrum(s, a, e, fs, file)
    return 0

def get_formant(wav_file_path, frames):

    # 音声をロード
    wav, fs = wavread(wav_file_path)
    formants = []

    for i in range(int(len(frames) / 2)):

        max_frame = float(frames[-1])

        center_frame = (float(frames[2 * i]) + float(frames[2 * i + 1])) / 2.0
        center = len(wav) * (center_frame / max_frame)

        # 音声波形の中心部分を切り出す
        # center = len(wav) / 2  # 中心のサンプル番号
        cuttime = 0.04         # 切り出す長さ [s]
        s = wav[int(center - cuttime / 2 * fs): int(center + cuttime / 2 * fs)]

        # t = np.arange(0.0, len(wav) / fs, 1/fs)

        # プリエンファシスフィルタをかける
        p = 0.97         # プリエンファシス係数
        s = preEmphasis(s, p)

        # ハミング窓をかける
        hammingWindow = np.hamming(len(s))
        s = s * hammingWindow

        # LPC係数を求める
        lpcOrder = 12
        r = autocorr(s, lpcOrder + 1)

        a, e = LevinsonDurbin(r, lpcOrder)

        # フォルマント検出( by Tasuku SUENAGA a.k.a. gunyarakun )

        # 根を求めて三千里
        rts = np.roots(a)
        # 共役解のうち、虚部が負のものは取り除く
        rts = np.array(list(filter(lambda x: np.imag(x) >= 0, rts)))

        # 根から角度を計算
        angz = np.arctan2(np.imag(rts), np.real(rts))
        # 角度の低い順にソート
        sorted_index = angz.argsort()
        # 角度からフォルマント周波数を計算
        freqs = angz.take(sorted_index) * (fs / (2 * np.pi))
        # 角度からフォルマントの帯域幅も計算
        bw = -1 / 2 * (fs / (2 * np.pi)) * np.log(np.abs(rts.take(sorted_index)))

        formantFreqs = []
        for i in range(len(freqs)):
            # フォルマントの周波数は90Hz超えで、帯域幅は400Hz未満
            if freqs[i] > 90 and bw[i] < 400:
                formantFreqs.append(freqs[i])
                # print("Formant Frequency - : %d" % freqs[i])

        """
        # 第2フォルマントまでとる
        if ((freqs[0] > 600 and freqs[0] < 1400) and (freqs[1] > 900  and freqs[1] < 2000)): result = "あ"
        elif((freqs[0] > 100 and freqs[0] < 410) and (freqs[1] > 1900 and freqs[1] < 3500)): result = "い"
        elif ((freqs[0] > 100 and freqs[0] < 700) and (freqs[1] > 1100 and freqs[1] < 2000)): result = "う"
        elif ((freqs[0] > 400 and freqs[0] < 800) and (freqs[1] > 1700 and freqs[1] < 3000)): result = "え"
        elif ((freqs[0] > 300 and freqs[0] < 900) and (freqs[1] > 500  and freqs[1] < 1300)): result = "お"
        else: result = "not match!"

        print("wave=" + wave_file + ": position=" + str(center_frame) + ": formant frequency -> 1: " + str(freqs[0]) + ", 2: " + str(freqs[1]) + ", result = '" + result + "'")
        """

        formants.append(freqs[0])
        formants.append(freqs[1])

    return formants


'''
if __name__ == "__main__":
    file = "a.wav"
    lpc_spectral_envelope(file)
    exit(0)
'''
