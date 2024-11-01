import numpy as np


def robust_min_max(img, consideration_factors=(0.1, 0.1)):
    # TODO: the 0.5 is used due to backwards compatibility.
    return np.quantile(img, [consideration_factors[0] * 0.5, 1 - consideration_factors[1] * 0.5])


def scale(img, old_range, new_range):

    scale = (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
    alpha = old_range[0]
    beta = new_range[0]

    img_new =  (img - alpha) * scale + beta
    return np.clip(img_new, new_range[0], new_range[1])

    # shift = -old_range[0] + new_range[0] * (old_range[1] - old_range[0]) / (new_range[1] - new_range[0])
    # scale = (new_range[1] - new_range[0]) / (old_range[1] - old_range[0])
    # return (img + shift) * scale


def scale_min_max(img, new_range):
    old_range = np.amin(img), np.amax(img)
    return scale(img, old_range, new_range)


def normalize_mr_robust(img, out_range=(-1, 1), consideration_factor=0.1):
    _, max = robust_min_max(img, (0, consideration_factor))
    old_range = (0, max)
    return scale(img, old_range, out_range)


def normalize(img, out_range=(-1, 1)):
    min_value = np.min(img)
    max_value = np.max(img)
    old_range = (min_value, max_value)
    return scale(img, old_range, out_range)


def normalize_robust(img, out_range=(-1, 1), consideration_factors=(0.1, 0.1)):
    min_value, max_value = robust_min_max(img, consideration_factors)
    if max_value == min_value:
        # fix to prevent div by zero
        max_value = min_value + 1
    old_range = (min_value, max_value)
    return scale(img, old_range, out_range)


def normalize_zero_mean_unit_variance(img):
    # add by zongkang, for spine MR segmentation 
    # need to convert img float16 to float64, otherwise std occur inf
    img = img.astype(np.float64)
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    img = img.astype(np.float16)
    return img
    # return (img - mean) / std

