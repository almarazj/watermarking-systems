import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.custom_loss import WatermarkLoss
from models.discriminator import Discriminator
from models.my_model import Model
import json
from utils.get_dataloader import get_loader

def train_epoch(
    trn_loader: DataLoader,
    model,
    discriminator,
    model_optimizer: torch.optim.Adam,
    disc_optimizer: torch.optim.Adam,
    device: torch.device):

    running_loss = 0
    num_total = 0.0
    model.train()
    discriminator.train()
    criterion = WatermarkLoss()
    
    for batch_x, batch_m in trn_loader:
        print(f"Batch {num_total}")
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_m = batch_m.to(device)
        
        # Encode
        batch_x_out = model.encode(batch_x, batch_m)
        batch_x_out = batch_x_out.to(device)
        # Decode
        batch_m_out = (model.decode(batch_x_out) >= 0.5).float().detach()
        batch_m_out = batch_m_out.to(device)
        
        # Discriminator
        real_preds = discriminator(batch_x.unsqueeze(1))
        fake_preds = discriminator(batch_x_out.unsqueeze(1))
        
        # Update model (generator)
        model_optimizer.zero_grad()
        disc_optimizer.zero_grad()
        
        # Compute loss
        total_loss  = criterion(batch_m, batch_x, batch_m_out, batch_x_out, real_preds, fake_preds)
        total_loss.backward()

        model_optimizer.step()
        disc_optimizer.step()        
   
        running_loss += total_loss.item() * batch_size
        print(f"Running loss: {running_loss}")
    running_loss /= num_total
    return running_loss

def train(loaders,
          model,
          discriminator,
          device,
          config):
    print("Start training...")
    train_losses = []
    model_optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    disc_optimizer = optim.Adam(discriminator.parameters(), lr=config["LEARNING_RATE"])
    for epoch in range(config["NUM_EPOCHS"]):
        print(f"Epoch: {epoch}")
        train_loss = train_epoch(loaders["train"], model, discriminator, model_optimizer, disc_optimizer, device)
        train_losses.append(train_loss)
        
        
def get_data_paths(conf_path: str) -> dict:
    # Read json file
    with open(conf_path, "r") as f_json:
        paths = json.loads(f_json.read())
    return paths
    
def main():
    
    config = {
        "BATCH_SIZE": 4,
        "LEARNING_RATE": 1e-4,
        "NUM_EPOCHS": 50, 
        "SEED": 1234,
        "NUM_POINTS": 16000,
        "NUM_BITS": 16,
        "FFT_SIZE": 1000,
        "HOP_LENGTH": 400,
        "NUM_LAYERS": 8
        }
    
    conf_path = "./datasets/paths.conf"
    data_paths = get_data_paths(conf_path)
    
    # 1.load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    discriminator = Discriminator().to(device)
    
    model = Model(num_point=config["NUM_POINTS"],
                  num_bit=config["NUM_BITS"],
                  n_fft=config["FFT_SIZE"],
                  hop_length=config["HOP_LENGTH"],
                  num_layers=config["NUM_LAYERS"]).to(device)
    
    loaders = get_loader(config["SEED"], data_paths, config["BATCH_SIZE"])
    
    train(loaders, model, discriminator, device, config)
    
if __name__ == '__main__':   
    main()