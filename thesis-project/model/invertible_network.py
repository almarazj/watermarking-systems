import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertibleBlock(nn.Module):
    def __init__(self):
        super(InvertibleBlock, self).__init__()
        self.phi = self._build_cnn_block()
        self.eta = self._build_cnn_block()
        self.rho = self._build_cnn_block()

    def _build_cnn_block(self):
        layers = []
        for _ in range(5):
            layers.append(nn.Conv2d(2, 2, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x, m):
        x_next = x + self.phi(m)
        m_next = m * torch.exp(self.rho(x_next)) + self.eta(x_next)
        return x_next, m_next

    def inverse(self, x_next, m_next):
        m = (m_next - self.eta(x_next)) * torch.exp(-self.rho(x_next))
        x = x_next - self.phi(m)
        return x, m

class InvertibleNetwork(nn.Module):
    def __init__(self, num_blocks=8):
        super(InvertibleNetwork, self).__init__()
        self.blocks = nn.ModuleList([InvertibleBlock() for _ in range(num_blocks)])

    def forward(self, x, m):
        for block in self.blocks:
            x, m = block(x, m)
        return x, m

    def inverse(self, x, m):
        for block in reversed(self.blocks):
            x, m = block.inverse(x, m)
        return x, m

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


class InvertibleNetworkLosses:
    def __init__(self, lambda_a=1.0, lambda_g=0.1):
        self.lambda_a = lambda_a
        self.lambda_g = lambda_g

    def message_loss(self, original_message, decoded_message):
        """
        Lm = ||mvec − mˆ vec||2^2
        """
        return F.mse_loss(decoded_message, original_message)

    def audio_loss(self, original_audio, watermarked_audio):
        """
        La = ||xwave − x'wave||2^2
        """
        return F.mse_loss(watermarked_audio, original_audio)

    def discriminator_loss(self, discriminator, original_audio, watermarked_audio):
        """
        Ld = log(1 − d(xwave)) + log(d(x'wave))
        """
        real_loss = torch.log(1 - discriminator(original_audio))
        fake_loss = torch.log(discriminator(watermarked_audio))
        return -(real_loss + fake_loss).mean()

    def generator_loss(self, discriminator, watermarked_audio):
        """
        Lg = log(1 − d(x'wave))
        """
        return -torch.log(1 - discriminator(watermarked_audio)).mean()

    def total_loss(self, original_audio, watermarked_audio, original_message, decoded_message, discriminator):
        """
        Ltotal = λa * La + Lm + λg * Lg
        """
        lm = self.message_loss(original_message, decoded_message)
        la = self.audio_loss(original_audio, watermarked_audio)
        lg = self.generator_loss(discriminator, watermarked_audio)
        
        return self.lambda_a * la + lm + self.lambda_g * lg

# Ejemplo de uso:
# losses = InvertibleNetworkLosses(lambda_a=1.0, lambda_g=0.1)
# Ltotal = losses.total_loss(original_audio, watermarked_audio, original_message, decoded_message, discriminator)
