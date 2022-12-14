from monai.networks.nets import BasicUNet
from monai.losses import DiceLoss
import numpy as np
from pytorch_lightning import LightningModule
import torch

class UNet_Train(LightningModule):
    def __init__(self, img_size=(1, 3, 96, 96), batch_size=8, lr=1e-4):
        super().__init__()

        self.save_hyperparameters()
        self.example_input_array = [torch.zeros(self.hparams.img_size)]

        self.model = BasicUNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            features=(64, 128, 256, 512, 1024, 128)
                                )

        self.DSC_loss = DiceLoss(include_background=False, sigmoid=True)

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, batch_idx, "val")

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #    return self(batch['image'])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def _prepare_batch(self, batch):
        return batch['image'], batch['segmentation']

    def _common_step(self, batch, batch_idx, stage: str):
        inputs, gt_input = self._prepare_batch(batch)
        outputs = self.forward(inputs)
        DSC_loss = self.DSC_loss(outputs, gt_input)
        train_steps = self.current_epoch + batch_idx

        self.log_dict({
            f'{stage}_DSC_loss': DSC_loss.item(),
            'step': float(train_steps),
            'epoch': float(self.current_epoch)}, batch_size=self.hparams.batch_size)

        if train_steps % 10 == 0:
            prediction = torch.round(torch.sigmoid(outputs))
            self.log_dict({
                'step': float(train_steps)}, batch_size=self.hparams.batch_size)
            self.logger.log_image(key="Ground Truth", images=[
                (gt_input.detach().cpu().numpy())[0, 0, :, :]],
                caption=["GT Segmentations"])
            self.logger.log_image(key="Input Image", images=[
                (inputs.detach().cpu().numpy())[0, 0, :, :]],
                caption=["Input Image"])
            self.logger.log_image(key="Prediction", images=[
                (prediction.detach().cpu().numpy())[0, 0, :, :]],
                caption=["Segmentations "])

        return DSC_loss

