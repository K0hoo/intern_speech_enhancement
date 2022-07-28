
from os.path import join
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn

# import model
from models import RealImg_Norm_LSTM
from execute import train, test

device = "cuda" if torch.cuda.is_available() else "cpu"
print("current device:", device)

# set hyper-parameters
num_epochs, batch_size, learning_rate, num_workers = 100, 32, 0.005, 4

# set model and other optimizer
model = RealImg_Norm_LSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, cooldown=0.9)
scaler = torch.cuda.amp.GradScaler()

def main(args):

    # set data_format
    data_format = {
        'transform': False,
        'validation_ratio': 5,
        'batch_size': 32
    }

    if args.continue_epoch:
        model_parameter = join(args.model_parameter, f"parameter_{args.continue_epoch}.pt")
        try:
            checkpoint = torch.load(model_parameter)
            assert(args.continue_epoch==int(checkpoint['epoch']))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            criterion.load_state_dict(checkpoint['criterion_state_dict'])
            if scheduler: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Model parameter with {args.continue_epoch} epoch(s) exists in file system.")
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
        continue_epoch=args.continue_epoch,
        num_epochs=num_epochs,
        data_format=data_format,
        num_workers=num_workers,
        device=device,
    )
    
    test(
        args=args,
        model=model,
        criterion=criterion,
        num_workers=num_workers,
        data_format=data_format,
        device=device
    )
