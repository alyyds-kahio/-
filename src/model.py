import torch
from torch import nn



class DoubleConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels
    ):

        super().__init__()


        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1
            ),

            nn.ReLU()

        )



    def forward(self,x):

        return self.block(x)




class UNet(nn.Module):


    def __init__(self):

        super().__init__()


        self.encoder1 = DoubleConv(
            1,
            64
        )


        self.pool = nn.MaxPool2d(2)


        self.encoder2 = DoubleConv(
            64,
            128
        )


        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                128,
                64,
                2,
                stride=2
            ),

            DoubleConv(
                64,
                64
            )

        )


        self.output = nn.Conv2d(
            64,
            1,
            1
        )



    def forward(self,x):


        x1 = self.encoder1(x)


        x2 = self.pool(x1)


        x2 = self.encoder2(x2)


        x3 = self.decoder(x2)


        out = self.output(x3)


        return torch.sigmoid(out)