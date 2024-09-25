import torch.nn as nn
import torch


class WatermarkLoss(nn.Module):
    def __init__(self, lambda_a=0.1, lambda_g=0.1):
        super(WatermarkLoss, self).__init__()
        self.lambda_a = lambda_a
        self.lambda_g = lambda_g
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, message, signal, noisy_signal, discriminator, encoder, decoder):
        # La: L2 loss between original audio and watermarked audio
        signal_wmd = encoder(signal, message)
        La = self.mse_loss(signal, signal_wmd)
        
        # Lm: L2 loss between original message and decoded message
        message_restored = decoder(noisy_signal)
        Lm = self.mse_loss(message, message_restored)

        # Ld: Discriminator loss (classifying original audio as 0, watermarked audio as 1)
        real_labels = torch.zeros(signal.size(0), 1)
        fake_labels = torch.ones(noisy_signal.size(0), 1)

        real_preds = discriminator(signal)
        fake_preds = discriminator(noisy_signal)
        Ld = self.bce_loss(real_preds, real_labels) + self.bce_loss(fake_preds, fake_labels)

        # Lg: Generator (encoder) tries to fool the discriminator
        Lg = self.bce_loss(discriminator(noisy_signal), real_labels)

        # Total loss: Ltotal = λa * La + Lm + λg * Lg
        Ltotal = self.lambda_a * La + Lm + self.lambda_g * Lg
        
        return Ltotal, Ld
