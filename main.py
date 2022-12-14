import os
from Data.Dataloader import Images
from Models.Training import UNet_Train

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    wandb_logger = WandbLogger(project="BIOE_658M_UNet")

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(dirpath="./saved_models/", save_top_k=1, monitor="val_DSC_loss", save_on_train_epoch_end=True)
    #checkpoint_path = "None"

    trainer = Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        devices=[0],
        max_epochs=1000,
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=1
    )

    trainer.fit(
        model=UNet_Train(),
        datamodule=Images(batch_size=20),
        #ckpt_path=checkpoint_path
    )
