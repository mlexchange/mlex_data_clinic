from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pl


class Optimizer(str, Enum):
    adadelta = "Adadelta"
    adagrad = "Adagrad"
    adam = "Adam"
    adamw = "AdamW"
    sparseadam = "SparseAdam"
    adamax = "Adamax"
    asgd = "ASGD"
    lbfgs = "LBFGS"
    rmsprop = "RMSprop"
    rprop = "Rprop"
    sgd = "SGD"


class Criterion(str, Enum):
    l1loss = "L1Loss"
    mseloss = "MSELoss"
    crossentropyloss = "CrossEntropyLoss"
    ctcloss = "CTCLoss"
    nllloss = "NLLLoss"
    poissonnllloss = "PoissonNLLLoss"
    gaussiannllloss = "GaussianNLLLoss"
    kldivloss = "KLDivLoss"
    bceloss = "BCELoss"
    bcewithlogitsloss = "BCEWithLogitsLoss"
    marginrankingloss = "MarginRankingLoss"
    hingeembeddingloss = "HingeEnbeddingLoss"
    multilabelmarginloss = "MultiLabelMarginLoss"
    huberloss = "HuberLoss"
    smoothl1loss = "SmoothL1Loss"
    softmarginloss = "SoftMarginLoss"
    multilabelsoftmarginloss = "MutiLabelSoftMarginLoss"
    cosineembeddingloss = "CosineEmbeddingLoss"
    multimarginloss = "MultiMarginLoss"
    tripletmarginloss = "TripletMarginLoss"
    tripletmarginwithdistanceloss = "TripletMarginWithDistanceLoss"


class TrainingParameters(BaseModel):
    target_size: tuple = Field(description='data target size')
    shuffle: bool = Field(description='shuffle data')
    batch_size: int = Field(description='batch size')
    val_pct: int = Field(description='validation percentage')
    latent_dim: int = Field(description='latent space dimension')
    base_channel_size: int = Field(description='number of base channels')
    num_epochs: int = Field(description='number of epochs')
    optimizer: Optimizer
    criterion: Criterion
    learning_rate: float = Field(description='learning rate')
    seed: Optional[int] = Field(description='random seed')


class EvaluationParameters(TrainingParameters):
    latent_dim: List[int] = Field(description='list of latent space dimensions')


class TestingParameters(BaseModel):
    target_size: tuple = Field(description='data target size')
    batch_size: int = Field(description='batch size')
    seed: Optional[int] = Field(description='random seed')


class Encoder(nn.Module):
    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 width: int,
                 height: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - width, height: Dimensionality of the input image
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        linear_dim = int(width * height / 64)
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * linear_dim * c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 width: int,
                 height: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - width, height: Dimensionality of the input image
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        self.width = width
        self.height = height
        c_hid = base_channel_size
        linear_dim = int(width * height / 64)
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * linear_dim * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            nn.Tanh()  # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, int(self.width / 8), int(self.height / 8))
        x = self.net(x)
        return x


class Autoencoder(pl.LightningModule):
    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 num_input_channels: int = 3,
                 optimizer: object = Optimizer,
                 criterion: object = Criterion,
                 encoder_class: object = Encoder,
                 decoder_class: object = Decoder,
                 learning_rate: float = 1e-3,
                 width: int = 32,
                 height: int = 32):
        super().__init__()
        self.optimizer = getattr(optim, optimizer.value)
        self.learning_rate = learning_rate
        if isinstance(criterion, Enum):
            criterion = getattr(nn, criterion.value)
            self.criterion = criterion()
        else:
            self.criterion = criterion
        self.train_loss = 0
        self.validation_loss = 0
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, width, height, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, width, height, latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width, height)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def _get_reconstruction_loss(self, batch):
        """
        Given a batch of images, this function returns the reconstruction loss (MSE in our case)
        """
        x, _ = batch  # We do not need the labels
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, x)
        #         loss = torch.nn.functional.mse_loss(x, x_hat, reduction="none")
        #         loss = loss.sum(dim=[1, 2, 3]).mean(dim=[0])
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.learning_rate)
        return {"optimizer": optimizer}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('train_loss', loss, on_epoch=True)
        self.train_loss += float(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('val_loss', loss, on_epoch=True)
        self.validation_loss += float(loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log('test_loss', loss, on_epoch=True)

    def on_train_epoch_end(self):
        current_epoch = self.current_epoch
        num_batches = self.trainer.num_training_batches
        train_loss = self.train_loss
        validation_loss = self.validation_loss
        self.train_loss = 0
        self.validation_loss = 0
        print(current_epoch, ' ', train_loss / num_batches, ' ', validation_loss, flush=True)

    def on_validation_epoch_end(self):
        num_batches = self.trainer.num_val_batches[0]  # may be a list[int]
        self.validation_loss = self.validation_loss / num_batches

    def on_train_end(self):
        print('Train process completed', flush=True)

