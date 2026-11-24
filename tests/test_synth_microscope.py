import numpy as np
from algae.sim.synth_microscope import make_image, CLASSES

def test_image_and_labels_are_consistent():
    rng = np.random.default_rng(0)
    img, labels = make_image(rng)
    assert img.shape == (640, 640, 3)
    assert sum(1 for l in labels if l[0] == 0) >= 30
    assert all(0 <= l[0] < len(CLASSES) for l in labels)
