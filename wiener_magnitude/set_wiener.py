
from os.path import join
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn

# import model
from models import Wiener_LSTM
from wiener_magnitude.exec_wiener import train, test

device = "cuda" if torch.cuda.is_available() else "cpu"

# set hyper-parameters
num_epochs, batch_size, learning_rate, validation_ratio, num_workers = 200, 64, 0.004, 5, 6

# set model and other optimizer
model = Wiener_LSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.2, patience=3, cooldown=2)
scaler = torch.cuda.amp.GradScaler()

def main(args):

    print(f"device: {device}")

    # set data_format
    data_format = {
        'mag_angle': True, # boolean value; true: magnitude-angle, false: real-img
        'logarithm': False, # boolean value; true: 2 seconds without zero padding
                            # boolean value; false: 4 seconds with zero padding
                            # logarithm means log-magnitude so, false transform and true logarithm is not allowed.
        'wiener': True, # wiener also means log-magnitude.
        'validation_ratio': validation_ratio, # int value; train:validation = <int value>-1:1
        'batch_size': batch_size # int value; batch size
    }

    assert(data_format['mag_angle'] or not data_format['logarithm'])
    assert(data_format['mag_angle'] or not data_format['wiener'])

    b_train = True
    output_checkpoint = join(args.root_folder, args.sub_folder)
    checkpoint_path = join(output_checkpoint, 'checkpoint')

    if args.continue_epoch:
        checkpoint_file = join(checkpoint_path, f"checkpoint_{args.continue_epoch}.pt")
        try:
            checkpoint = torch.load(checkpoint_file)
            # the input argument about start epoch should be same as the start epoch in parameter.
            assert(args.continue_epoch==int(checkpoint['epoch']))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            criterion.load_state_dict(checkpoint['criterion_state_dict'])
            if scheduler: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Model checkpoint with {args.continue_epoch} epoch(s) exists in file system.")
            if args.continue_epoch == num_epochs: b_train = False
        except:
            print(f"There is no such checkpoint with {args.continue_epoch} epoch(s).")
    else:
        print("There is no saved checkpoint.")

    if b_train:
        print("The model will be trained.")
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
    
    print("The model will be tested.")
    test(
        args=args,
        model=model,
        criterion=criterion,
        num_workers=num_workers,
        data_format=data_format,
        device=device
    )
