import cv2
import numpy as np
import torch
from torch import Tensor
from torchvision import transforms


def calc_psnr(image1: Tensor, image2: Tensor):
    to_image = transforms.ToPILImage()
    image1 = cv2.cvtColor(
        (np.array(to_image(image1))).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    image2 = cv2.cvtColor(
        (np.array(to_image(image2))).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    return cv2.PSNR(image1, image2)


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience  # 設定ストップカウンタ
        self.counter = 0  # 現在のカウンタ値
        self.best_score = None  # ベストスコア
        self.early_stop = False  # ストップフラグ
        self.val_loss_min = np.Inf  # 前回のベストスコア記憶用

    def __call__(self, val_loss, model, device):
        """
        特殊(call)メソッド
        実際に学習ループ内で最小lossを更新したか否かを計算させる部分
        """
        if val_loss >= self.val_loss_min:  # 最小ロスを更新できなかった場合
            self.counter += 1  # ストップカウンタを+1
            if (
                self.counter >= self.patience
            ):  # 設定カウントを上回ったらストップフラグをTrueに変更
                self.early_stop = True
        else:  # 最小ロスを更新した場合
            # モデル生成
            # torch.save(model.state_dict(), "model.pth")
            model.to(torch.device("cpu"))
            dummy_input = torch.randn(1, 3, 128, 128, device="cpu")
            torch.onnx.export(
                model,
                dummy_input,
                "submit/model.onnx",
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {2: "height", 3: "width"}},
            )
            model.to(device)
            self.val_loss_min = val_loss  # その時のlossを記録する
            self.counter = 0  # ストップカウンタリセット
