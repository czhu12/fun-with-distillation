import torch.optim.lr_scheduler as lr_scheduler
import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.metrics.functional import accuracy


class BaseImageClassificationModel(pl.LightningModule):
    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        predictions = self.net(x)
        return predictions
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.net(x)
        loss = F.cross_entropy(output, y)
        preds = torch.argmax(output, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.net.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return [optimizer], [scheduler]

    def prepare_data(self):
        CIFAR10(root=self.data_dir, train=True, download=True)
        CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(root=self.data_dir, train=True, transform=self.transform_train)
            self.cifar_train, self.cifar_val = random_split(cifar_full, [45000, 5000])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(root=self.data_dir, train=False, transform=self.transform_test)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=32)

    @property
    def transform_train(self):
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    @property
    def transform_test(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])