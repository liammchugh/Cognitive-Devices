import torch, torchaudio
from torch import Tensor
from pathlib import Path
from torchvision.transforms.functional import resize
import os

# if "KMP_DUPLICATE_LIB_OK" in os.environ:
#     del os.environ["KMP_DUPLICATE_LIB_OK"]

class AccelToRGBMel:
    """
    Convert 3-axis accelerometer waveform → (3, H, W) log-mel image.
    Suitable for CLIP/BLIP/VLM fine-tuning.
    Correct STFT params for 64 Hz data.
        Log-compress + [0, 1] normalise.
        Square resize to match common VLM backbones.
        Vectorised mel-spectrogram for speed.
        Graceful CPU fallback.
    """
    def __init__(
        self,
        sample_rate: int = 64,                 # ← 64 Hz after resampling
        win_len_sec: float = 4.0,              # window ≈ n_fft / sr
        hop_frac: float = 0.25,                # 75 % overlap
        n_mels: int = 128,
        img_size: int = 224,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        n_fft = int(2 ** torch.ceil(torch.log2(torch.tensor(sample_rate * win_len_sec))))
        hop_length = int(n_fft * hop_frac)

        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        ).to(device)

        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80).to(device)
        self.img_size = img_size
        self.device = device

    @torch.inference_mode()
    def __call__(self, accel: Tensor) -> Tensor:
        """
        accel : (3, T) float tensor on **any** device
        returns: (3, img_size, img_size) float32 in [0,1]
        """
        accel = accel.to(self.device, dtype=torch.float32)

        # Vectorised: (3, T) → (3, n_mels, frames)
        m = self.melspec(accel)
        m = self.to_db(m)                      # log-scale

        # Normalise each channel separately to [0,1]
        m_min, m_max = m.amin(dim=(1,2), keepdim=True), m.amax(dim=(1,2), keepdim=True)
        m = (m - m_min) / (m_max - m_min + 1e-8)

        # Resize spectrogram (H=n_mels, W=frames) to square image
        m = resize(m.unsqueeze(0), size=(self.img_size, self.img_size), antialias=True).squeeze(0)

        return m                               # ready for VLM encoder


if __name__ == "__main__":
    sr = 64                     # after your signal-prep stage
    sl = 30               # 30 seconds of data
    dummy = torch.randn(3, sr*sl)   # 30-second sample

    xfm = AccelToRGBMel(sample_rate=sr)
    rgb_img = xfm(dummy)        # (3, 224, 224), float32, [0..1]

    # quick sanity-plot
    import matplotlib.pyplot as plt
    duration_sec = dummy.shape[1] / sr
    plt.imshow(
        rgb_img[0].cpu(),
        extent=(0, duration_sec, 0, rgb_img.shape[1]),
        origin="lower",
        aspect="auto"
    )
    plt.title("Axis-0 log-mel (normalised)")
    plt.xlabel("Time (s)")

    # debug only
    import psutil, re, os
    def list_omp_libs():
        proc = psutil.Process()
        return sorted({m.path for m in proc.memory_maps()
                    if re.search(r"(?:lib|i)?omp|gomp", os.path.basename(m.path), re.I)})
    # print("Loaded OpenMP libraries →", list_omp_libs())

    # debug only
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    plt.show()

