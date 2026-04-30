import numpy as np
import torch
import matplotlib.pyplot as plt
import pypulseq as pp
import MRzeroCore as mr0
import torchkbnufft as tkbn

# Load sequence
mr_seq = mr0.Sequence.import_file("gre.seq")

signal,_ = mr0.util.simulate("gre.seq", None)

signal = signal.numpy()
kspace = signal.reshape(128, 128)

image = np.fft.fftshift(
    np.fft.ifft2(
        np.fft.ifftshift(kspace)
    )
)

plt.figure()
plt.imshow(np.abs(image), cmap='gray')
plt.title("Reconstructed GRE Image")
plt.show()