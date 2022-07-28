
from os.path import join, exists
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn

from models import SimpleLSTM
from execute import train, test

device = "cuda" if torch.cuda.is_available() else "cpu"
print("current device:", device)

# setting hyper-parameters
num_epochs, batch_size, learning_rate, num_workers = 100, 32, 0.002, 4

# setting model and other optimizer
model = SimpleLSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, cooldown=2)
scaler = torch.cuda.amp.GradScaler()

def main(args):
    
    data_format = {
        'transform': True,
        'validation_ratio': 5,
        'batch_size': 32
    }

    if args.continue_epoch:
        model_parameter = join(args.model_parameter, f"parameter_{args.continue_epoch}.pt")
        try:
            model.load_state_dict(torch.load(model_parameter))
            print(f"Model parameter with {args.model_parameter} epoch(s) exists in file system.")
        except:
            print(f"There is no such parameter with {args.continue_epoch} epoch(s).")
    else:
        print("There is no saved parameter. The model will be trained.")
    
    train(
        args=args, 
        model=model, 
        criterion=criterion, 
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        num_epochs=num_epochs,
        data_format=data_format,
        num_workers=num_workers,
        device=device
    )
    
    test(
        args=args,
        model=model,
        criterion=criterion,
        num_workers=num_workers,
        data_format=data_format,
        device=device
    )