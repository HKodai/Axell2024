# ハイパーパラメータ
batch_size = 8
num_workers = 16
num_epoch = 10000
learning_rate = 2e-4
step_size = 50
gamma = 0.5
patience = 10

import torch
from torch import nn, clip, tensor
from torch.utils import data
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import sys

sys.path.append(os.getcwd())
from lib.dataprocess import get_dataset
from lib.util import calc_psnr, EarlyStopping

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# モデルの定義
class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1
        )
        self.prelu = nn.PReLU()

        self.conv_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(self, X: tensor) -> tensor:
        X_shortcut = X
        X = self.prelu(self.conv_1(X))
        X = self.conv_2(X)
        X_out = X + X_shortcut
        return X_out


class EDSR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = 4
        self.conv_1 = nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=3, padding=1
        )
        self.prelu1 = nn.PReLU()

        self.res_blocks = nn.ModuleList([ResBlock(64, 64) for _ in range(6)])
        self.prelus = nn.ModuleList([nn.PReLU() for _ in range(6)])

        self.conv_2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.prelu2 = nn.PReLU()

        self.conv_3 = nn.Conv2d(
            in_channels=64,
            out_channels=(3 * self.scale * self.scale),
            kernel_size=3,
            padding=1,
        )

        self.pixel_shuffle = nn.PixelShuffle(self.scale)

    def forward(self, X: tensor) -> tensor:
        X = self.prelu1(self.conv_1(X))
        X_shortcut = X
        for res_block, prelu in zip(self.res_blocks, self.prelus):
            X = prelu(res_block(X))
        X = self.conv_2(X)
        X += X_shortcut
        X = self.prelu2(X)
        X = self.conv_3(X)
        X = self.pixel_shuffle(X)
        X_out = clip(X, 0.0, 1.0)
        return X_out


def train():
    model = EDSR()
    model.to(device)
    # 実行ファイルの名前を取得（拡張子を除く）
    file_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    log_dir = "./log"
    # ./log以下に実行ファイル名と同じ名前のディレクトリを作成
    target_dir = os.path.join(log_dir, file_name)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    writer = SummaryWriter(target_dir)

    # データセットの取得
    train_dataset, validation_dataset = get_dataset()
    train_data_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    validation_data_loader = data.DataLoader(
        validation_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )

    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size, gamma)
    l2 = MSELoss()
    earlystopping = EarlyStopping(patience)

    for epoch in range(num_epoch):
        try:
            # 学習
            model.train()
            train_loss = 0.0
            validation_loss = 0.0
            train_psnr = 0.0
            validation_psnr = 0.0
            for _, (low_resolution_image, high_resolution_image) in tqdm(
                enumerate(train_data_loader),
                desc=f"EPOCH[{epoch}] TRAIN",
                total=len(train_data_loader),
                leave=False,
            ):
                low_resolution_image = low_resolution_image.to(device)
                high_resolution_image = high_resolution_image.to(device)
                optimizer.zero_grad()
                output = model(low_resolution_image)
                loss = l2(output, high_resolution_image)
                loss.backward()
                train_loss += loss.item() * low_resolution_image.size(0)
                for image1, image2 in zip(output, high_resolution_image):
                    train_psnr += calc_psnr(image1, image2)
                optimizer.step()
            scheduler.step()

            # 検証
            model.eval()
            with torch.no_grad():
                for _, (low_resolution_image, high_resolution_image) in tqdm(
                    enumerate(validation_data_loader),
                    desc=f"EPOCH[{epoch}] VALIDATION",
                    total=len(validation_data_loader),
                    leave=False,
                ):
                    low_resolution_image = low_resolution_image.to(device)
                    high_resolution_image = high_resolution_image.to(device)
                    output = model(low_resolution_image)
                    loss = l2(output, high_resolution_image)
                    validation_loss += loss.item() * low_resolution_image.size(0)
                    for image1, image2 in zip(output, high_resolution_image):
                        validation_psnr += calc_psnr(image1, image2)
            writer.add_scalar("train/loss", train_loss / len(train_dataset), epoch)
            writer.add_scalar("train/psnr", train_psnr / len(train_dataset), epoch)
            writer.add_scalar(
                "validation/loss", validation_loss / len(validation_dataset), epoch
            )
            writer.add_scalar(
                "validation/psnr", validation_psnr / len(validation_dataset), epoch
            )
            # writer.add_image("output", output[0], epoch)

            earlystopping(validation_loss / len(validation_dataset), model, device)
            if earlystopping.early_stop:
                break

        except Exception as ex:
            print(f"EPOCH[{epoch}] ERROR: {ex}")

    writer.close()


if __name__ == "__main__":
    train()
