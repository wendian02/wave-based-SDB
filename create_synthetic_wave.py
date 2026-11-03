"""生成已知水深的解析传播波，供 spatial_dft_wlai.ipynb 验证。

运行：python create_synthetic_wave.py
输出：脚本旁 data/ 中的多帧 GeoTIFF 和时间/真值 JSON。
这是亮度代理的理想平面波，不是真实卫星光学成像或 SWASH 仿真。
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import numpy as np
import rasterio
from rasterio.transform import from_origin


def main():
    # 先独立指定水深与波长，再用色散关系求频率；不使用反演结果构造数据。
    depth, wavelength, direction = 10.0, 80.0, 30.0
    gravity, resolution, size, frame_count = 9.81, 2.0, 301, 30
    center_x, center_y = 451.0, 499.0
    frame_interval, brightness_amplitude, noise_std = 1.0, 100.0, 0.5
    k = 2 * np.pi / wavelength
    omega = np.sqrt(gravity * k * np.tanh(k * depth))
    period, speed = 2 * np.pi / omega, omega / k
    axis = (np.arange(size) - size // 2) * resolution
    x, y = np.meshgrid(axis, -axis)  # 列向东、行向南；y 坐标正向北。
    theta = np.deg2rad(direction)
    spatial_phase = k * (x * np.cos(theta) + y * np.sin(theta)) + 0.2

    output = Path(__file__).resolve().parent / "data" / "synthetic_wave_depth10m.tif"
    output.parent.mkdir(exist_ok=True)
    transform = from_origin(center_x - size * resolution / 2,
                            center_y + size * resolution / 2, resolution, resolution)
    rng = np.random.default_rng(42)
    with rasterio.open(output, "w", driver="GTiff", width=size, height=size,
                       count=frame_count, dtype="float32", crs="EPSG:32632",
                       transform=transform, compress="deflate", predictor=3) as dst:
        for frame in range(1, frame_count + 1):
            time = (frame - 1) * frame_interval
            image = (1000 + brightness_amplitude * np.cos(spatial_phase - omega * time)
                     + rng.normal(0, noise_std, size=(size, size)))
            dst.write(image.astype("float32"), frame)
            dst.set_band_description(frame, f"Synthetic brightness at t={time:g} s")
        dst.update_tags(DATA_TYPE="analytic travelling plane wave brightness proxy")

    start = datetime(2000, 1, 1, tzinfo=timezone.utc)
    metadata = {
        "DATA_TYPE": "synthetic analytic wave; not observed satellite data",
        "FRAMES_TIME": {
            str(frame): (start + timedelta(seconds=(frame - 1) * frame_interval)).isoformat()
            for frame in range(1, frame_count + 1)
        },
        "GROUND_TRUTH": {
            "depth_m": depth, "wavelength_m": wavelength, "direction_deg": direction,
            "period_s": float(period), "celerity_m_s": float(speed),
            "gravity_m_s2": gravity, "resolution_m": resolution,
            "center_xy": [center_x, center_y], "frame_interval_s": frame_interval,
            "brightness_amplitude": brightness_amplitude, "noise_std": noise_std,
            "random_seed": 42,
            "model": "I(x,y,t) = 1000 + A cos(k(x cos(theta) + y sin(theta)) - omega t + 0.2) + noise",
            "dispersion": "omega^2 = g k tanh(k h); k = 2 pi / wavelength",
        },
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved: {output}")
    print(f"Depth={depth:g} m, wavelength={wavelength:g} m, direction={direction:g} deg")
    print(f"Period={period:.6f} s, speed={speed:.6f} m/s")
    print(f"Phase shift for frames 10/13: {omega * 3:.6f} rad (< pi)")


if __name__ == "__main__":
    main()
