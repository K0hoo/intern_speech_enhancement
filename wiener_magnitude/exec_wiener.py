
import torch
import torch.nn as nn
from os.path import join

from pesq import pesq

from dataset import cal_istft, save_result

sampling_rate = 16000

def train(
    args, 
    # deep learning components
    model,
    criterion,
    optimizer,
    scheduler,
    scaler,
    # deep learning parameter
    continue_epoch,
    num_epochs,
    device,
    # data loader
    train_loader,
    validation_loader,
    # tensor board
    writer
):

    amp_on = True if args.fp16 and scaler else False
    output_folder = join(args.root_folder, args.sub_folder)
    output_train_log = join(output_folder, 'train_log.txt')
    ouptut_checkpoint = join(output_folder, 'checkpoint')


    for epoch in range(continue_epoch, num_epochs):
        
        # 1. Train
        print(f'Train: {epoch + 1}/{num_epochs} is on.')
        train_loss = 0.0
        model.train()
        for i, (noisy, clean, noise) in enumerate(train_loader):
            
            noisy_mag = noisy['mag'].to(device)
            clean_mag = clean['mag'].to(device)
            noise_mag = noise['mag'].to(device)
            target = ((clean_mag ** 2) / (clean_mag ** 2 + noise_mag ** 2 + 1e-8)).to(device=device)

            with torch.cuda.amp.autocast(amp_on):
                output = model(noisy_mag)
                loss = criterion(output, target)
            
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

            train_loss += loss.item()
            
            if (i + 1) % 100 == 0:
                print(f'Train Epoch: {epoch + 1}/{num_epochs}, Batch step: {(i + 1)}/{train_loader.__len__()}, Loss: {train_loss/100:.6f}')
                writer.add_scalar(
                    'train loss', 
                    train_loss/100, 
                    epoch * train_loader.__len__() + i + 1
                )
                train_loss = 0.0

        # 2. Validation
        print(f'Validation: {epoch + 1}/{num_epochs} is on.')
        validation_loss = 0
        with torch.no_grad():
            model.eval()
            for i, (noisy, clean, noise) in enumerate(validation_loader):

                noisy_mag = noisy['mag'].to(device)
                clean_mag = clean['mag'].to(device)
                noise_mag = noise['mag'].to(device)
                target = ((clean_mag ** 2) / (clean_mag ** 2 + noise_mag ** 2 + 1e-8)).to(device=device)

                with torch.cuda.amp.autocast(amp_on):
                    output = model(noisy_mag)
                    loss = criterion(output, target)

                validation_loss += loss.item()

        validation_loss /= validation_loader.__len__()

        # 3. Validation loss and learning rate is printed.
        message = f"Epoch: {epoch + 1}/{num_epochs}, validation loss: {validation_loss:.6f}, learning rate: {str(optimizer.param_groups[0]['lr'])}"
        train_log_file = open(output_train_log, 'a', newline='')
        train_log_file.write(message + '\n')
        train_log_file.close()
        print(message)

        writer.add_scalar(
            'validation loss', 
            validation_loss,
            epoch + 1
        )

        # 4. Learning rate is updated.
        if scheduler:
            scheduler.step(validation_loss)

        # 5. The parameter is saved.
        name = f"checkpoint_{epoch + 1}.pt"
        parameter_path = join(ouptut_checkpoint, name)
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
            
            
def test(
    args, 
    # deep learning components
    model, 
    criterion, 
    # deep learning parameters
    device,
    # data loader
    seen_test_loader,
    unseen_test_loader,
    # data foramt
    mag_angle
):
    
    # Some parameter is defined.
    bl_seen = [True, False]
    amp_on = True if args.fp16 else False
    output_folder = join(args.root_folder, args.sub_folder)
    output_test_log = join(output_folder, 'test_log.txt')
    output_result = join(output_folder, 'test_result')
    output_seen_result = join(output_result, 'seen')
    output_unseen_result = join(output_result, 'unseen')

    with torch.no_grad():
        model.eval()
        for b_seen in bl_seen:
            total_test_loss = 0
            total_pesq_score= 0
            test_loader = seen_test_loader if b_seen else unseen_test_loader

            # The batch size for test is just 1. noisy, clean, and target data is just for 1.
            # clean data is just for making result data.
            for i, (noisy, clean, noise) in enumerate(test_loader):

                # 1. The output is calculated.
                noisy_mag, clean_mag, noise_mag = noisy['mag'].to(device), clean['mag'], noise['mag']
                target = ((clean_mag ** 2) / (clean_mag ** 2 + noise_mag ** 2 + 1e-8)).to(device)

                with torch.cuda.amp.autocast(amp_on):
                    output = model(noisy_mag)
                    test_loss = criterion(output, target).item()

                total_test_loss += test_loss

                # 2. When the number of data is the multiple of 100, the sound is saved.
                if (i + 1) % 100 == 0:
                    result_root_path = output_seen_result if b_seen else output_unseen_result
                    file_name = ("s_result" if b_seen else "u_result") + str(i + 1) + ".wav"
                    
                    output = output.type(torch.DoubleTensor)
                    noisy_mag = noisy_mag.to('cpu')

                    # 3. The amplitude is made from STFT form.
                    # When output is mask, the stft_mag is output(mask) * noisy.
                    esti_amp = cal_istft(stft_mag=(output * noisy_mag), stft_angle=noisy['angle'], mag_angle=mag_angle).to('cpu')
                    noisy_amp = cal_istft(stft_mag=noisy_mag, stft_angle=noisy['angle'], mag_angle=mag_angle).to('cpu')
                    clean_amp = cal_istft(stft_mag=clean_mag, stft_angle=clean['angle'], mag_angle=mag_angle).to('cpu')

                    # 4. pesq
                    pesq_score = pesq(sampling_rate, clean_amp.numpy()[0], esti_amp.numpy()[0], 'wb')
                    total_pesq_score += pesq_score


                    # 5. The created amplitude is saved by save_result function.
                    # If the sound is saved successfully, it returns True.
                    if save_result(output_test_log, result_root_path, file_name, esti_amp, noisy_amp, clean_amp):
                        message = f"{file_name} is created with {test_loss:.6f} loss. pesq: {pesq_score}"
                        print(message)
                        test_log_file = open(output_test_log, 'a', newline='')
                        test_log_file.write(message + '\n')
                        test_log_file.close()

            # 6. The test loss is printed.
            total_test_loss /= test_loader.__len__()
            total_pesq_score /= (test_loader.__len__()//100)
            message = f"Seen test loss: {total_test_loss}" if b_seen else f"Unseen test loss: {total_test_loss}, pesq score: {pesq_score/(test_loader.__len__()//100)}"
            message += f" pesq: {total_pesq_score}"
            print(message)
            test_log_file = open(output_test_log, 'a', newline='')
            test_log_file.write(message + '\n')
            test_log_file.close()
