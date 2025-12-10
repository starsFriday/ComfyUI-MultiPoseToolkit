import numpy as np
import cv2
import warnings


def _gaussian_blur(heatmaps, kernel=11):
    radius = kernel // 2
    size = 2 * radius + 1
    for n in range(heatmaps.shape[0]):
        for k in range(heatmaps.shape[1]):
            heatmaps[n][k] = cv2.GaussianBlur(heatmaps[n][k], (size, size), 0)
    return heatmaps


def _get_max_preds(heatmaps):
    N, K, H, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((N, K, 1))
    idx = idx.reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = (preds[:, :, 0]) % W
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / W)

    pred_mask = np.tile((maxvals > 0.0), (1, 1, 2))
    preds *= pred_mask.astype(np.float32)
    return preds, maxvals


def _taylor(heatmap, coord):
    px = int(np.floor(coord[0]))
    py = int(np.floor(coord[1]))
    if 1 < px < heatmap.shape[1] - 2 and 1 < py < heatmap.shape[0] - 2:
        derivative = np.zeros((2, 1))
        derivative[0] = 0.5 * (heatmap[py][px + 1] - heatmap[py][px - 1])
        derivative[1] = 0.5 * (heatmap[py + 1][px] - heatmap[py - 1][px])
        hessian = np.zeros((2, 2))
        hessian[0][0] = heatmap[py][px + 1] - 2 * heatmap[py][px] + heatmap[py][px - 1]
        hessian[1][1] = heatmap[py + 1][px] - 2 * heatmap[py][px] + heatmap[py - 1][px]
        hessian[0][1] = 0.25 * (
            heatmap[py + 1][px + 1] - heatmap[py - 1][px + 1] - heatmap[py + 1][px - 1] + heatmap[py - 1][px - 1])
        hessian[1][0] = hessian[0][1]
        # add light regularization so nearly-flat heatmaps do not yield a singular Hessian
        hessian += np.eye(2, dtype=hessian.dtype) * 1e-6
        try:
            hessian = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            hessian = np.linalg.pinv(hessian)
        offset = -hessian @ derivative
        coord += offset.reshape(2)
    return coord


def transform_preds(coords, center, scale, output_size, use_udp=False):
    assert coords.shape[1] in (2, 4, 5)
    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]
    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5
    return target_coords


def keypoints_from_heatmaps(heatmaps,
                            center,
                            scale,
                            unbiased=False,
                            post_process='default',
                            kernel=11,
                            use_udp=False):
    heatmaps = heatmaps.copy()
    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert post_process != 'megvii'

    if post_process is False:
        warnings.warn('post_process=False is deprecated, please use post_process=None', DeprecationWarning)
        post_process = None
    elif post_process is True:
        post_process = 'unbiased' if unbiased else 'default'
    elif post_process == 'default' and unbiased:
        post_process = 'unbiased'

    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    preds, maxvals = _get_max_preds(heatmaps)

    if post_process == 'unbiased':
        heatmaps = np.log(np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
        for n in range(N):
            for k in range(K):
                preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
    elif post_process is not None:
        for n in range(N):
            for k in range(K):
                heatmap = heatmaps[n][k]
                px = int(preds[n][k][0])
                py = int(preds[n][k][1])
                if 1 < px < W - 1 and 1 < py < H - 1:
                    diff = np.array([
                        heatmap[py][px + 1] - heatmap[py][px - 1],
                        heatmap[py + 1][px] - heatmap[py - 1][px]
                    ])
                    preds[n][k] += np.sign(diff) * .25
                    if post_process == 'megvii':
                        preds[n][k] += 0.5

    for i in range(N):
        preds[i] = transform_preds(preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5

    return np.concatenate([preds, maxvals], axis=2)
