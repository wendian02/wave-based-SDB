

# %%
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from scipy.signal import find_peaks, fftconvolve
from skimage.transform import radon

class WaveBased:
    def __init__(self, tiles, res, delta_t): # tile 2x600x600 or other size
        self.tiles = tiles
        self.smooth_halfsize = 10        # 背景平滑核的半宽，单位：像元；0 表示关闭。 kernel size
        self.dx = res # resolution
        self.delta_t = delta_t

        self.q_min = 0.001
        self.q_step = 0.0001                  # 空间频率步长，单位：cycles/m。
        self.direction_prominence = 0.3 # 峰保留到最大峰的0.3
        self.refinement_prominence = 0.1
        self.angle_halfwidth = 10        # 在候选方向左右各搜索 10°
        self.g = 9.81
        self.candidates = []
        self.refinements = []

        self.depth_min = 0.5 # min depth

        self.T_min = 3.0 
        self.T_max = 25.0  # 最终估计周期的筛选范围，单位：秒。
        

    def detrend_plane(self, image):
        # plane bg
        y, x = np.indices(image.shape)
        A = np.column_stack([x.ravel(), y.ravel(), np.ones(image.size)]) # [x1,y1,1] x [a,b,c].T=ax1+by1+c
        plane = (A @ np.linalg.lstsq(A, image.ravel(), rcond=None)[0]).reshape(image.shape) # lstsq-> least squares 
        return image - plane

    def preprocess(self, image, halfsize):
        # 1 detrend plane bg
        image = self.detrend_plane(image)

        if halfsize > 0:
            y, x = np.ogrid[:2 * halfsize + 1, :2 * halfsize + 1]
            radius = np.hypot(y - halfsize - 1, x - halfsize - 1)
            kernel = np.cos(np.pi * radius / (2 * np.sqrt(2) * halfsize))
            background = fftconvolve(np.pad(image, halfsize, mode="edge"),
                                    kernel / kernel.sum(), mode="valid")
            image = image - background  # 2 detrend curve bg, kernel size = halfsize
            image = self.detrend_plane(image) # 3 detrend plane bg

        return image
    
    def cross_spectrum(self, C1, C2):
        cross = C1 * C2.conj() # C2 取共轭
        phase = np.angle(cross)
        return cross, phase, np.abs(cross) * np.abs(phase)


    def periodic_peaks(self, values, prominence):
        count = len(values)
        peaks, _ = find_peaks(np.tile(values, 3), prominence=prominence) # 0 & 180 的时候可能有峰，所以要复制3份，取中间的.peak是对应的横坐标，也就是角度
        return peaks[(peaks >= count) & (peaks < 2 * count)] - count
    
    def dft_at_frequencies(self, signals, frequencies, dx):
        positions = np.arange(signals.shape[0]) * dx
        return np.exp(-2j * np.pi * frequencies[:, None] * positions) @ signals
    


    def wave_based(self): # tile windows
        self.candidates = []
        self.refinements = []
        tiles = self.tiles
        dx = self.dx
        n = tiles.shape[1] # tile size

        # get circle shape
        yy, xx = np.ogrid[:n, :n]
        r = n // 2
        disk = (yy - r)**2 + (xx - r)**2 <= r**2

        # preprocess
        tiles = np.array([self.preprocess(tile, self.smooth_halfsize) * disk for tile in tiles])

        # radon transform
        directions = np.arange(180, dtype=float) # RT 0-180 degree
        s1, s2 = [radon(tile, theta=directions, circle=True, preserve_range=True) for tile in tiles]

        q_fft = np.fft.rfftfreq(n, d=dx) # rfftfreq 自动计算采样frequency，
        C1 = np.fft.rfft(s1, axis=0)
        C2 = np.fft.rfft(s2, axis=0)
        cross, phase, score = self.cross_spectrum(C1, C2)
        direction_score = score.max(axis=0) # direction

        max_direction_score = direction_score.max()
        if not np.isfinite(max_direction_score) or max_direction_score <= 0: # if no corse angle
            return
        direction_score /= max_direction_score # norm direction scoure
        # coarse direction
        peaks = self.periodic_peaks(direction_score, self.direction_prominence) # 峰保留到最大峰的0.3

        # fine direction
        q_max = 1 / (2 * dx)
        # 多个方向，每个方向取±10度，分别算每个度的 max score，得到这个方向的fine direction, 并且采样频率q_fine设置得更精细，之前
        q_fine = np.arange(self.q_min, q_max, self.q_step) # fine frequency

        for peak in peaks:
            columns = (peak + np.arange(-self.angle_halfwidth, self.angle_halfwidth + 1)) % len(directions) # 限定在0-180，超过取模，-1 % 180-> 179 
            F1 = self.dft_at_frequencies(s1[:, columns], q_fine, dx) # q_fine下的傅里叶数
            F2 = self.dft_at_frequencies(s2[:, columns], q_fine, dx)# q_fine下的傅里叶数
            _, fine_phase, fine_score = self.cross_spectrum(F1, F2) # C1C2*
            angle_score = fine_score.max(axis=0) # keep max score
            max_angle_score = angle_score.max()
            if not np.isfinite(max_angle_score) or max_angle_score <= 0: # if no angle
                continue
            angle_peaks, _ = find_peaks(angle_score / max_angle_score, prominence=self.refinement_prominence) # 0.1 prominence
            
            # 得fine direction的max frequency，保存到candidates (refinement记录±10个方向的所有值)
            frequency_peaks = fine_score[:, angle_peaks].argmax(axis=0) # 
            refinement = dict(
                directions=directions[columns], # 本轮的 21 个方向
                C1=F1, # 频率数 × 21
                C2=F2, 
                phase=fine_phase,
                score=fine_score)

            # 这个peak下面得到的fine angle和frequency都放到candidates，比如corse angle是20±10，那这个candidates可能会直接增加31,35,37
            for qi, di in zip(frequency_peaks, angle_peaks): 
                self.candidates.append(
                    dict(
                        projection_deg=directions[columns[di]],
                        q=q_fine[qi],
                        phase=fine_phase[qi, di],
                        score=fine_score[qi, di],
                        refinement=len(self.refinements), # cor refinement id
                        qi=qi,
                        di=di))
                        
            self.refinements.append(refinement)

    def get_depth(self):
        self.depth = np.nan
        self.wave_based()
        candidates = self.candidates

        rows = []
        for candidate in candidates:
            q, phase_shift = candidate["q"], candidate["phase"]
            omega = abs(phase_shift / self.delta_t)
            k = 2 * np.pi * q
            T = 2 * np.pi / omega if omega > 0 else np.inf # 
            speed = omega / k 
            gamma = omega**2 / (self.g * k) # gamma = omega^2 / gk
            depth = np.arctanh(gamma) / k if 0.1 < gamma < 1 else np.nan # Bergsma 2021 RSE 0.3-0.9, S2Shores use 0.1 -1
   
            direction = (candidate["projection_deg"] + (180 if phase_shift / self.delta_t < 0 else 0)) % 360

            rows.append(dict(
                direction_deg=direction,
                from_north_deg=(270 - direction) % 360,
                wavelength_m=1/q,
                phase_shift_rad=phase_shift,
                period_s=T,
                celerity_m_s=speed,
                gamma=gamma,
                depth_m=depth,
                score=candidate["score"],
                valid_physics=bool(
                    np.isfinite(depth) 
                    and depth > self.depth_min
                    and self.T_min <= T <= self.T_max))) # 
        if len(rows) != 0:
            results = (pd.DataFrame(rows).sort_values("score", ascending=False)
                    .drop_duplicates(["direction_deg", "wavelength_m"]).reset_index(drop=True))


            valid_results = results.loc[results["valid_physics"]]

            depth_weighted = (
                np.average(
                    valid_results["depth_m"],
                    weights=valid_results["gamma"],
                )
                if not valid_results.empty
                else np.nan)
            self.depth = depth_weighted
        return self.depth





if __name__ == "__main__":

    data_source = "synthetic"   # "synthetic"：已知水深验证；"swash"：原 SWASH 数据。
    input_files = {
        "synthetic": Path("./data/synthetic_wave_depth10m.tif"),
        "swash": Path("/Users/wendy/Downloads/SWASH_8_2/testcase_8_2_cropped.tif"),
    }
    product_path = input_files[data_source]
    frames = (10, 13)
    center_xy = (451.0, 499.0)
    window_m = 400.0

    with product_path.with_suffix(".json").open() as file:
        metadata = json.load(file)
    times = [datetime.fromisoformat(metadata["FRAMES_TIME"][str(frame)]) for frame in frames]
    delta_t = (times[1] - times[0]).total_seconds()

    print(f'delta_t = {delta_t}')

    with rasterio.open(product_path) as src:
        dx, dy = src.res
        row, col = src.index(*center_xy)
        half = min(int(window_m / (2 * dx)), row, src.height - 1 - row, col, src.width - 1 - col)
        n = 2 * half + 1
        window = Window(col - half, row - half, n, n)
        raw = src.read(frames, window=window, masked=True).astype(float).filled(np.nan)
        actual_center = src.xy(row, col)
    # raw -> 2x LxL

    print(WaveBased(raw, dx, delta_t).get_depth())



