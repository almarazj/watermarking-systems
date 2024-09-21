import torch.nn as nn
import torch


class WatermarkLoss(nn.Module):
    def __init__(self, lambda_a=0.1, lambda_g=0.1):
        super(WatermarkLoss, self).__init__()
        self.lambda_a = lambda_a
        self.lambda_g = lambda_g
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, mvec, x_wave, x_wave_prime, discriminator, encoder, decoder):
        # Lm: L2 loss between original message and decoded message
        mvec_hat = decoder(x_wave_prime)
        Lm = self.mse_loss(mvec, mvec_hat)

        # La: L2 loss between original audio and watermarked audio
        x_wave_encoded = encoder(x_wave, mvec)
        La = self.mse_loss(x_wave, x_wave_encoded)

        # Ld: Discriminator loss (classifying original audio as 0, watermarked audio as 1)
        real_labels = torch.zeros(x_wave.size(0), 1)
        fake_labels = torch.ones(x_wave_prime.size(0), 1)

        real_preds = discriminator(x_wave)
        fake_preds = discriminator(x_wave_prime)
        Ld = self.bce_loss(real_preds, real_labels) + self.bce_loss(fake_preds, fake_labels)

        # Lg: Generator (encoder) tries to fool the discriminator
        Lg = self.bce_loss(discriminator(x_wave_prime), real_labels)

        # Total loss: Ltotal = λa * La + Lm + λg * Lg
        Ltotal = self.lambda_a * La + Lm + self.lambda_g * Lg
        
        return Ltotal, Ld
