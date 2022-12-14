from Data.Data import ImageData
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import LightningDataModule

#STOP! NO NEED TO DO ANYTHING WITH THIS FILE!
#You may adjust this file if you would like to change the training validation split.

class Images(LightningDataModule):
    def __init__(self, batch_size: int = None):
        super().__init__()
        scans = ImageData()

        self.train, self.val = random_split(scans, [int(len(scans) * 0.8), len(scans) - int(len(scans) * 0.8)])
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, drop_last=True, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, drop_last=True, pin_memory=True, persistent_workers=True)

    #def predict_dataloader(self):
    #    return DataLoader(self.test, batch_size=self.batch_size, num_workers=16)

