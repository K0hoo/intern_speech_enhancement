
import torch
import torch.nn as nn
import torchaudio
from os.path import join

from dataset import get_test_dataset, get_train_dataset, cal_istft

sampling_rate = 16000

def train(args, model, criterion, optimizer, scheduler=None, scaler=None, continue_epoch=0, num_epochs=100, data_format=None, num_workers=4, device="cuda"):

    train_loader, validation_loader = get_train_dataset(
        root_folder=args.data_root_folder,
        transform=data_format['transform'],
        logarithm=data_format['logarithm'],
        validation_ratio=data_format['validation_ratio'],
        batch_size=data_format['batch_size'],
        num_workers=num_workers
    )

    amp_on = True if args.fp16 and scaler else False
    print("amp on" if amp_on else "amp off")

    for epoch in range(continue_epoch, num_epochs):
        
        # Train
        model.train()
        for i, (noisy, clean) in enumerate(train_loader):
            
            noisy, clean = noisy.to(device), clean.to(device)

            with torch.cuda.amp.autocast(amp_on):
                output = model(noisy)
                loss = criterion(output, clean)
            
            if amp_on:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            if amp_on:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            if (i + 1) % 10 == 0:
                message = f'Train Epoch: {epoch + 1}/{num_epochs}, Batch Step: {i + 1}/{train_loader.__len__()}, Loss: {loss.item():.6f}'
                print(message)

        # Validation
        validation_loss = 0
        with torch.no_grad():
            model.eval()
            for i, (noisy, clean) in enumerate(validation_loader):

                noisy, clean = noisy.to(device), clean.to(device)

                with torch.cuda.amp.autocast(amp_on):
                    output = model(noisy)
                    loss = criterion(output, clean)

                validation_loss += loss.item()

                if (i + 1) % 10 == 0:
                    message = f'Validation Epoch: {epoch + 1}/{num_epochs}, Batch Step: {i + 1}/{validation_loader.__len__()}, Loss: {loss.item():.6f}'
                    print(message)

            validation_loss /= validation_loader.__len__()

        message = f"Epoch: {epoch + 1}/{num_epochs}, validation loss: {validation_loss:.6f}, learning rate: {str(optimizer.param_groups[0]['lr'])}"
        print(message)
        train_log_file = open(args.output_train_log, 'a', newline='')
        train_log_file.write(message + '\n')
        train_log_file.close()

        if scheduler:
            scheduler.step(validation_loss)

        if (epoch + 1) % 5 == 0:
            name = f"parameter_{epoch + 1}.pt"
            parameter_path = join(args.model_parameter, name)
            if scheduler:
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "criterion_state_dict": criterion.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()
                }, parameter_path)
            else:
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "criterion_state_dict": criterion.state_dict()
                }, parameter_path)
            
            
def test(args, model, criterion, num_workers=4, data_format=None, device="cuda"):
    
    seen_test_loader, unseen_test_loader = get_test_dataset(
        root_folder=args.data_root_folder,
        transform=data_format['transform'],
        logarithm=data_format['logarithm'],
        num_workers=num_workers
    )
    bl_seen = [True, False]
    amp_on = True if args.fp16 else False
    transform = data_format['transform']

    with torch.no_grad():
        model.eval()
        for b_seen in bl_seen:
            total_test_loss = 0
            test_loader = seen_test_loader if b_seen else unseen_test_loader
            for i, (noisy, clean) in enumerate(test_loader):
                noisy, clean = noisy.to(device), clean.to(device)

                with torch.cuda.amp.autocast(amp_on):
                    output = model(noisy)
                    test_loss = criterion(output, clean).item()
                    
                total_test_loss += test_loss

                if (i + 1) % 100 == 0:
                    result_root_path = args.test_seen_result if b_seen else args.test_unseen_result
                    file_name = ("s_result" if b_seen else "u_result") + str(i + 1) + ".wav"
                    
                    output = torch.tensor(output, dtype=torch.float32)
                    esti_amp = cal_istft(stft=output, transform=transform).to('cpu')
                    noisy_amp = cal_istft(stft=noisy, transform=transform).to('cpu')
                    clean_amp = cal_istft(stft=clean, transform=transform).to('cpu')

                    if save_result(args, result_root_path, file_name, esti_amp, noisy_amp, clean_amp):
                        message = f"{file_name} is created with {test_loss:.6f} loss."
                        print(message)
                        test_log_file = open(args.output_test_log, 'a', newline='')
                        test_log_file.write(message)
                        test_log_file.close()

            total_test_loss /= test_loader.__len__()
            message = f"Seen test loss: {total_test_loss}" if b_seen else f"Unseen test loss: {total_test_loss}"
            print(message)
            test_log_file = open(args.output_test_log, 'a', newline='')
            test_log_file.write(message + '\n')
            test_log_file.close()


def save_result(args, root_path, file_name, esti_amp, noisy_amp, clean_amp):
    
    try:
        result_file = join(join(root_path, "result"), file_name)
        noisy_file = join(join(root_path, "noisy"), file_name)
        clean_file = join(join(root_path, "clean"), file_name)

        mag_factor = (0.9 / torch.max(esti_amp))
        esti_amp *= mag_factor

        torchaudio.save(
            result_file,
            torch.unsqueeze(esti_amp, 0),
            sampling_rate
        )

        torchaudio.save(
            noisy_file,
            torch.unsqueeze(noisy_amp, 0),
            sampling_rate
        )

        torchaudio.save(
            clean_file,
            torch.unsqueeze(clean_amp, 0),
            sampling_rate
        )

        return True
    except Exception as e:
        test_log_file = open(args.output_test_log, 'a', newline='')
        test_log_file.write(f"Exception when creating reuslt file. {e}\n")
        test_log_file.close()
        return False