# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import tempfile
import time

import numpy as np
import torch

from PIL import Image

from cog import BasePredictor, Input, Path
from torchvision import transforms

from salient_cropper import SalientCropper
from utils.data_process import preprocess_img, postprocess_img


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        self.device = device

        torch_dtype = torch.float16
        print(f"Using torch_dtype: {torch_dtype}")
        self.torch_dtype = torch_dtype

        flag = 0  # 0 for TranSalNet_Dense, 1 for TranSalNet_Res

        if flag:
            from TranSalNet_Res import TranSalNet

            model = TranSalNet()
            model.load_state_dict(torch.load("pretrained_models/TranSalNet_Res.pth"))
        else:
            from TranSalNet_Dense import TranSalNet

            model = TranSalNet()
            model.load_state_dict(torch.load("pretrained_models/TranSalNet_Dense.pth"))

        model = model.to(device=device, dtype=torch_dtype)
        model.eval()
        self.model = model

        self.crop = SalientCropper()

    def predict(
        self,
        image: Path = Input(description="The image to crop from."),
        # scale: float = Input(
        #     description="Factor to scale image by", ge=0, le=10, default=1.5
        # ),
        width: int = Input(description="Desired crop width.", ge=0),
        height: int = Input(description="Desired crop height.", ge=0),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(image)
        # output = self.model(processed_image, scale)
        # return postprocess(output)

        print(f"image: {image}")

        # with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=True if self.device == "cuda" else False):
        with torch.inference_mode():
            tic = time.time()
            # Obtain saliency map
            img = preprocess_img(
                str(image)
            )  # padding and resizing input image into 384x288
            img = np.array(img) / 255.0
            img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
            img = torch.from_numpy(img)
            img = img.to(device=self.device, dtype=self.torch_dtype)
            pred_saliency = self.model(img)
            toPIL = transforms.ToPILImage()
            pic = toPIL(pred_saliency.squeeze())
            pred_saliency = postprocess_img(
                pic, str(image)
            )  # restore the image to its original size as the result
            tac = time.time()

            # Salient crop
            crop_outs = self.crop(
                img=np.array(Image.open(image)),
                sal_map=pred_saliency,
                width=width,
                height=height,
            )
            cropped = crop_outs[0]
            toe = time.time()

            print(f"Saliency prediction time : {tac - tic:.1f} s")
            print(f"Salient crop time        : {toe - tac:.1f} s")

            assert cropped.shape[:2] == (height, width)

            output_path = Path(tempfile.mkdtemp()) / "cropped.jpg"
            Image.fromarray(cropped).save(output_path, quality=95)
            return output_path
