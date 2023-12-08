from pprint import pprint
from typing import List

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def resize_image_to_minimum_width_and_height(
    img: np.ndarray,
    width: int,
    height: int,
    interpolation=cv2.INTER_CUBIC
):
    ori_h, ori_w = img.shape[:2]
    max_ratio = max(width / ori_w, height / ori_h)
    target_w = int(max_ratio * ori_w)
    target_h = int(max_ratio * ori_h)
    # print(f"target_w: {target_w}")
    # print(f"target_h: {target_h}")

    if not (ori_w, ori_h) == (target_w, target_h):
        img = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    return img


class SalientCropper(object):
    def __init__(self):
        pass

    def _preprocess_inputs(
        self, img: np.ndarray, sal_map: np.ndarray, width: int, height: int
    ):
        # Ensure image sides is large enough to crop
        img = resize_image_to_minimum_width_and_height(
            img=img, width=width, height=height
        )

        # Ensure saliency map has the same size as the image
        if sal_map.shape[:2] != img.shape[:2]:
            sal_map = cv2.resize(
                sal_map, img.shape[:2][::-1], interpolation=cv2.INTER_LINEAR
            )

        return img, sal_map

    def __call__(
        self,
        img: np.ndarray,
        sal_map: np.ndarray,
        width: int,
        height: int,
        cropping_window_multiplier: str = "uniform",
        allow_zoom: bool = False,
        scale_range=np.arange(0.2, 1.0, 0.2).tolist()
        + [1.0, 1.25, 1.5, 1.75, 2.0, 4.0],
        verbose: bool = True,
    ):
        """Perform salient crop on an image.

        Args:
            img (np.ndarray): Image array.
            sal_map (np.ndarray): Saliency map.
            width (int): Crop width.
            height (int): Crop height.
            cropping_window_multiplier (str, optional): Cropping window score multiplier type. Defaults to "rule_of_thirds".
            allow_zoom (bool, optional): Whether zooming is allowed. Defaults to True.
            scale_range (List[float], optional): If allow_zoom=True, the range of scale that is allowed to be performed on the image. Defaults to np.arange(0.1, 1.0 + 0.1, 0.1).tolist().
            verbose (bool, optional): Defaults to True.

        Returns:
            [(np.ndarray, float, bool)]: Tuple of cropped image array, crop_parameters, score, and boolean indicating whether crop is valid.
        """

        img, sal_map = self._preprocess_inputs(
            img=img, sal_map=sal_map, width=width, height=height
        )

        h, w = img.shape[:2]
        num_steps = 32

        if allow_zoom:
            # Perform salient crop at various scales and return the best crop

            crops = []
            crop_params = []
            scores = []
            valids = []
            scales = []
            for s in scale_range:
                target_h = int(h * s)
                target_w = int(w * s)

                if target_h < height or target_w < width:
                    if verbose:
                        print(
                            f"(scale: {s}) Skipping... target image dimension(s) smaller than crop dimension(s)."
                        )
                    continue

                stride_h = max(target_h // num_steps, 1)
                stride_w = max(target_w // num_steps, 1)
                stride = (stride_h, stride_w)

                if target_h != h or target_w != w:
                    if verbose:
                        print(
                            f"(scale: {s}) Resizing image from ({w}*{h}) to ({target_w} * {target_h})"
                        )
                    img_resized = cv2.resize(
                        img, (target_w, target_h), interpolation=cv2.INTER_CUBIC
                    )
                    sal_map_resized = cv2.resize(
                        sal_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    img_resized = img
                    sal_map_resized = sal_map

                cropped, params, score, valid = self._salient_crop(
                    img_resized,
                    sal_map_resized,
                    width,
                    height,
                    stride=stride,
                    cropping_window_multiplier=cropping_window_multiplier,
                )
                params["scale"] = s

                # print(f"cropped: {cropped.shape}")
                crops.append(cropped)
                crop_params.append(params)
                scores.append(score)
                valids.append(valid)
                scales.append(s)

            if verbose:
                pprint(f"shapes: {[c.shape for c in crops]}")
                pprint(f"crop_params: {crop_params}")
                pprint(f"scores: {scores}")
                pprint(f"valids: {valids}")
                pprint(f"scales: {scales}")

            # Sort first by crop validity, then by score, then by scale.
            # sorted_indices = np.argsort(scores)[::-1]
            sorted_indices = np.lexsort((scales, scores, valids))[::-1]
            if verbose:
                pprint(f"sorted scores: {np.array(scores)[sorted_indices].tolist()}")
                pprint(f"sorted valids: {np.array(valids)[sorted_indices].tolist()}")
                pprint(f"sorted scales: {np.array(scales)[sorted_indices].tolist()}")

            best_index = sorted_indices[0]
            if verbose:
                print(f"best_index: {best_index}")
            return (
                crops[best_index],
                crop_params[best_index],
                scores[best_index],
                valids[best_index],
            )
        else:
            stride_h = max(h // num_steps, 1)
            stride_w = max(w // num_steps, 1)
            stride = (stride_h, stride_w)

            return self._salient_crop(
                img,
                sal_map,
                width,
                height,
                stride=stride,
                cropping_window_multiplier=cropping_window_multiplier,
            )

    def _salient_crop(
        self,
        img: np.ndarray,
        sal_map: np.ndarray,
        width: int,
        height: int,
        stride: int = 1,
        cropping_window_multiplier: str = "rule_of_thirds",
        verbose: bool = True,
    ):
        """Perform saliency crop by solving the following optimization problem
        using brute-force sliding window technique with 2d-convolution.

            argmax ( (f_score(sal_map) * weights).sum() ) for c in C,

            where:
                sal_map is the saliency map
                f_score is a function that maps sal_map to an array of the same shape
                weights is an array having the shape of the cropping window for spatial pixel score adjustment
                C is the set of valid crops,
                * is the Hadamard product (element wise multiplication)

        Args:
            img (np.ndarray): Image array.
            sal_map (np.ndarray): Saliency map.
            width (int): Crop width.
            height (int): Crop height.
            stride (int, optional): Sliding window stride. Defaults to 1.
            cropping_window_multiplier (str, optional): Cropping window score multiplier type. Defaults to "rule_of_thirds".
            verbose (bool, optional): Defaults to True.

        Returns:
            [(np.ndarray, float, bool)]: Tuple of cropped image array, crop parameters, score, and boolean indicating whether crop is valid.
        """

        if verbose:
            print(f"Image size - width, height: {img.shape[1]}, {img.shape[0]}")
            print(f"Crop size  - width, height: {width}, {height}")
        assert (
            img.shape[:2] == sal_map.shape[:2]
        ), "Image and saliency map should have the same size."

        if not isinstance(stride, (list, tuple)):
            stride = (stride, stride)

        # Get score map from saliency map, this can be a non-identity function
        score_map = sal_map

        # Convert score map to tensor
        score_map_t = TF.to_tensor(score_map)
        assert score_map_t.ndim == 3
        score_map_t = score_map_t.unsqueeze(0)

        # This is our sliding window
        w = self._get_weights(width, height, mode=cropping_window_multiplier)

        with torch.inference_mode():
            out = F.conv2d(score_map_t, w, padding=0, stride=stride)

            print(f"stride: {stride}")
            print(f"out.shape: {out.shape}")
            print(f"score_map_t.shape: {score_map_t.shape}")

            out = F.interpolate(
                out,
                (int(out.size(-2) * stride[0]), int(out.size(-1) * stride[1])),
                mode="bilinear",
            )
            assert out.size(-2) < score_map_t.size(-2)
            assert out.size(-1) < score_map_t.size(-1)

            if out.shape[-2:] != score_map_t.shape[-2:]:
                h_diff = score_map_t.size(-2) - out.size(-2)
                w_diff = score_map_t.size(-1) - out.size(-1)
                padding_left = w_diff // 2
                padding_right = w_diff - padding_left
                padding_top = h_diff // 2
                padding_bottom = h_diff - padding_top
                padding = (padding_left, padding_right, padding_top, padding_bottom)
                out = F.pad(out, padding, mode="constant", value=0)
                assert out.shape[-2:] == score_map_t.shape[-2:]

            out = torch.squeeze(out).numpy()
            print(f"padded out.shape: {out.shape}")

        # Normalize as ratio over score_map sum to handle comparison over multiple scales
        out = out / (score_map.sum() + 1e-8)
        # assert out.min() >= 0.0
        # assert out.max() <= 1.0

        # The location of the highest value pixel in the output map is the crop center
        best_score = np.max(out)
        indices = np.argwhere(out == best_score)  # (n, 2)
        assert indices.ndim == 2 and indices.shape[-1] == 2

        # Flat score map
        assert out.ndim == 2
        if len(indices) == out.shape[0] * out.shape[1]:
            crop_center = (img.shape[0] // 2, img.shape[1] // 2)
        else:
            crop_center = indices[0]

        y, x = crop_center
        y -= height // 2
        x -= width // 2
        x = max(0, x)
        y = max(0, y)

        if verbose:
            print(f"Crop params: x, y, width, height: {x}, {y}, {width}, {height}")

        cropped = self.crop(img, x, y, width, height)
        crop_params = {"x": x, "y": y, "width": width, "height": height, "scale": 1.0}

        return cropped, crop_params, best_score, cropped.shape[:2] == (height, width)

    def crop(
        self, image_array: np.ndarray, x: int, y: int, w: int, h: int
    ) -> np.ndarray:
        """Crop an image given its position, and crop width and height.

        Args:
            arr (np.ndarray): Image array.
            x (int): Image top left x-coordinate.
            y (int): Image top left y-coordinate.
            w (int): Crop width.
            h (int): Crop height.

        Returns:
            [np.ndarray]: Cropped image array.
        """

        assert image_array.ndim in [2, 3], "Image array should be 2 or 3 dimensional."

        if image_array.ndim == 3:
            cropped = image_array[y : y + h, x : x + w, :]
        else:
            cropped = image_array[y : y + h, x : x + w]

        return cropped

    def _get_weights(
        self, width: int, height: int, mode: str = "rule_of_thirds"
    ) -> torch.Tensor:
        """Generate a weights matrix of shape (1, 1, width, height).
        This weights is used for location based scoring much like an archer target board.

        Args:
            width (int): Width.
            height (int): Height.

        Returns:
            [torch.Tensor]: Weight matrix
        """

        _mode = mode.lower().strip()
        if _mode == "uniform":
            # Uniform weights
            return torch.ones(1, 1, height, width)
        elif _mode == "rule_of_thirds":
            w = np.ones(shape=(height, width), dtype=np.float32)

            # w[int(height * 0.33/2):int(height * (1-0.33/2)),
            #   int(width * 0.33/2):int(width * (1-0.33/2))] = 1.5

            # 'Rule of thirds' photography rule for good image composition
            w[
                int(height * 0.33 - height * 0.075) : int(
                    height * 0.33 + height * 0.075
                ),
                :,
            ] += 1.0
            w[
                int(height * 0.67 - height * 0.075) : int(
                    height * 0.67 + height * 0.075
                ),
                :,
            ] += 1.0
            w[
                :, int(width * 0.33 - width * 0.075) : int(width * 0.33 + width * 0.075)
            ] += 1.0
            w[
                :, int(width * 0.67 - width * 0.075) : int(width * 0.67 + width * 0.075)
            ] += 1.0
            w = np.clip(w, 0, 2)
            # plt.imshow(w)
            return torch.from_numpy(w.reshape(1, 1, height, width))
        else:
            raise ValueError(f"Unknown mode: {mode}")
