
from os.path import join
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset import get_train_dataset, get_test_dataset

from models_TCN import CRM_TCN
from crm_complex.execute import train, test

device = "cuda" if torch.cuda.is_available() else "cpu"

# set hyper-parameters
continue_epoch = 300 # int value; the epoch that training is resumed
num_epochs = 300 # int value; the epoch that training is ended
batch_size = 64
learning_rate = 0.005
validation_ratio = 5 # int value; train:validation = <int value>-1:1
num_workers = 6

causal = False
input_size = 514 # frequency
hidden_size = 512
num_channels = [hidden_size, hidden_size, hidden_size, input_size]
kernel_size = 3

model = CRM_TCN(input_size, input_size, num_channels, kernel_size, causal).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, cooldown=2)
scaler = torch.cuda.amp.GradScaler()

# set data_format
mag_angle = False # boolean value; true: magnitude-angle, false: real-img
logarithm = False # boolean value; true: 2 seconds without zero padding
                # boolean value; false: 4 seconds with zero padding
                # logarithm means log-magnitude so, false transform and true logarithm is not allowed.
wiener = False # wiener also means log-magnitude.

def set(args):

    print(f"device: {device}")

    writer = SummaryWriter(f'{args.target_type}/{args.model}/runs')

    assert(mag_angle or not logarithm)
    assert(mag_angle or not wiener)

    param = sum(p.numel() for p in model.parameters())
    print(f'The number of parameters: {param}')

    b_train = args.train
    output_folder = join(args.root_folder, args.target_type, args.model)
    checkpoint_path = join(output_folder, 'checkpoint')

    if continue_epoch:
        checkpoint_file = join(checkpoint_path, f"checkpoint_{continue_epoch}.pt")
        try:
            checkpoint = torch.load(checkpoint_file)
            # the input argument about start epoch should be same as the start epoch in parameter.
            assert(continue_epoch==int(checkpoint['epoch']))
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            criterion.load_state_dict(checkpoint['criterion_state_dict'])
            if scheduler: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Model checkpoint with {continue_epoch} epoch(s) exists in file system.")
            if continue_epoch == num_epochs: b_train = False
        except:
            print(f"There is no such checkpoint with {continue_epoch} epoch(s).")
            return
    else:
        print("There is no saved checkpoint.")

    if b_train:

        print("The model will be trained.")

        # load data for train and validation
        train_loader, validation_loader = get_train_dataset(
            root_folder=args.data_root_folder,
            transform={
                'mag_angle': mag_angle,
                'logarithm': logarithm
            },
            validation_ratio=validation_ratio,
            batch_size=batch_size,
            num_workers=num_workers
        )

        train_args = {
            'args': args,
            'model': model, 'criterion': criterion, 'optimizer': optimizer, 'scheduler': scheduler, 'scaler': scaler,
            'continue_epoch': continue_epoch, 'num_epochs': num_epochs, 'device': device,
            'train_loader': train_loader, 'validation_loader': validation_loader,
            'writer': writer
        }

        train(**train_args)

        writer.flush()
    
    if args.test:

        print("The model will be tested.")

        # load data for test
        seen_test_loader, unseen_test_loader = get_test_dataset(
            root_folder=args.data_root_folder,
            transform={
                'mag_angle': mag_angle,
                'logarithm': logarithm
            },
            num_workers=num_workers
        )        

        test_args = {
            'args': args,
            'model': model, 'criterion': criterion,
            'device': device,
            'seen_test_loader': seen_test_loader, 'unseen_test_loader': unseen_test_loader,
            'mag_angle': mag_angle
        }

        test(**test_args)

    writer.close()
    print("done.")
