import torch.nn as nn
import torch


class WatermarkLoss(nn.Module):
    def __init__(self, lambda_a=0.1, lambda_g=0.1):
        super(WatermarkLoss, self).__init__()
        self.lambda_a = lambda_a
        self.lambda_g = lambda_g
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def forward(self, message, signal, message_restored, signal_wmd, real_preds, fake_preds):
        # La: L2 loss between original audio and watermarked audio
        La = self.mse_loss(signal, signal_wmd)
        
        # Lm: L2 loss between original message and decoded message
        Lm = self.mse_loss(message, message_restored)

        # Ld: Discriminator loss (classifying original audio as 0, watermarked audio as 1)
        real_labels = torch.zeros(signal.size(0), 1).to(real_preds.device)
        fake_labels = torch.ones(signal_wmd.size(0), 1).to(fake_preds.device)

        Ld = self.bce_loss(real_preds, real_labels) + self.bce_loss(fake_preds, fake_labels)

        # Lg: Generator (encoder) tries to fool the discriminator
        Lg = self.bce_loss(fake_preds, real_labels)

        # Total loss: Ltotal = λa * La + Lm + λg * Lg
        Ltotal = self.lambda_a * La + Lm + self.lambda_g * Lg
        
        return Ltotal + Ld
