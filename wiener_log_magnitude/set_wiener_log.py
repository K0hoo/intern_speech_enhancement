
from os.path import join
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset import get_test_dataset, get_train_dataset

# import model
from models import Wiener_LSTM
from wiener_log_magnitude.exec_wiener_log import train, test

device = "cuda" if torch.cuda.is_available() else "cpu"

# set hyper-parameters
continue_epoch = 0
num_epochs = 200
batch_size = 64
learning_rate = 0.005
validation_ratio = 5
num_workers = 6

# set model and other optimizer
model = Wiener_LSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=0.5, patience=3, cooldown=2)
scaler = torch.cuda.amp.GradScaler()

# set data_format
mag_angle = True, # boolean value; true: magnitude-angle, false: real-img
logarithm = True, # boolean value; true: 2 seconds without zero padding
                # boolean value; false: 4 seconds with zero padding
                # logarithm means log-magnitude so, false transform and true logarithm is not allowed.
wiener = True, # wiener also means log-magnitude.

def main(args):

    print(f"device: {device}")

    writer = SummaryWriter(f'{args.sub_folder}/runs')

    assert(mag_angle or not logarithm)
    assert(mag_angle or not wiener)

    b_train = args.train
    output_folder = join(args.root_folder, args.sub_folder)
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

        dataiter = iter(train_loader)
        (noisy, _, _) = dataiter.next()
        writer.add_graph(model, noisy['mag'].to(device))

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
