# In this file, the functions are set up for training the different networks
# Models are always loaded from the latest epoch and saved each epoch, to account for interrupted training. 

def fit_convnet(train_data_loader, conv_model, loss_fn, conv_optimizer, device, epochs, validation_data_loader, train_loss, validation_loss, train_accuracy, validation_accuracy):
    # Load the model from the latest trained epoch  
    with open("/content/gdrive/My Drive/colab/models/pcam_conv_last_epoch.txt") as f:
        start_epoch = int(f.readlines()[0])
    
    if start_epoch > 0:
        load_model_path = "/content/gdrive/My Drive/colab/models/pcam_conv_epoch_" + str(start_epoch)
        conv_model.load_state_dict(torch.load(load_model_path))


    # Keep track of the best performance on the validation set
    with open("/content/gdrive/My Drive/colab/models/pcam_conv_best_val_loss.txt") as f:
      best_v_loss = float(f.readlines()[0])

    # Train untill the required epoch
    print("Convnet")
    for t in range(start_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # Decrease the learning rate and weight decay after training for 5 and 10 epochs
        if t >=5:
          conv_optimizer = torch.optim.Adam(conv_model.parameters(), lr = 1e-4, weight_decay = 1e-5)

        if t >=10:
          conv_optimizer = torch.optim.Adam(conv_model.parameters(), lr = 1e-5, weight_decay = 1e-6)

        # Train a single epoch
        t_loss, t_accuracy = train_epoch(train_data_loader, conv_model, loss_fn, conv_optimizer, device)    
        
        # Save model parameters
        model_path = "/content/gdrive/My Drive/colab/models/pcam_conv_epoch_" + str(t+1)
        torch.save(conv_model.state_dict(), model_path)

        with open("/content/gdrive/My Drive/colab/models/pcam_conv_last_epoch.txt", 'w') as f:
            f.write(str(t+1))

        # Check performance on the validation set
        v_loss, v_accuracy = validation(validation_data_loader, conv_model, loss_fn, device)

        # Save training and validation set performance
        train_loss.loc[t, 'convnet'] = t_loss
        validation_loss.loc[t, 'convnet'] = v_loss
        train_accuracy.loc[t, 'convnet'] = t_accuracy
        validation_accuracy.loc[t, 'convnet'] = v_accuracy
        train_loss.to_csv("/content/gdrive/My Drive/colab/results/train_loss.csv")
        validation_loss.to_csv("/content/gdrive/My Drive/colab/results/validation_loss.csv")
        train_accuracy.to_csv("/content/gdrive/My Drive/colab/results/train_accuracy.csv")
        validation_accuracy.to_csv("/content/gdrive/My Drive/colab/results/validation_accuracy.csv")
        
        # Save state dict of best performing epoch
        if v_loss < best_v_loss:
          best_v_loss = v_loss
          model_path = "/content/gdrive/My Drive/colab/models/pcam_conv_best_epoch"
          torch.save(conv_model.state_dict(), model_path)
          with open("/content/gdrive/My Drive/colab/models/pcam_conv_best_val_loss.txt", 'w') as f:
            f.write(str(best_v_loss))

def fit_ft_resnet(train_data_loader, ft_resnet50, loss_fn, ft_resnet50_optimizer, device, epochs, restart, validation_data_loader, train_loss, validation_loss, train_accuracy, validation_accuracy):
    # Load the model from the latest trained epoch  
    with open("/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_last_epoch.txt") as f:
        start_epoch = int(f.readlines()[0])
    
    if start_epoch > 0:
        load_model_path = "/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_epoch_" + str(start_epoch)
        ft_resnet50.load_state_dict(torch.load(load_model_path))

    # Keep track of the best performance on the validation set
    with open("/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_best_val_loss.txt") as f:
      best_v_loss = float(f.readlines()[0])  

    # Train untill the required epoch
    print("Finetune ResNet50")
    for t in range(start_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # Decrease the learning rate and weight decay after training for 5 and 10 epochs
        if t >=5:
          ft_resnet50_optmizer = torch.optim.Adam(ft_resnet50.fc.parameters(), lr = 1e-4, weight_decay = 1e-5)

        if t >=10:
          ft_resnet50_optimizer = torch.optim.Adam(ft_resnet50.fc.parameters(), lr = 1e-5, weight_decay = 1e-6)
        
        # Train a single epoch
        t_loss, t_accuracy = train_epoch(train_data_loader, ft_resnet50, loss_fn, ft_resnet50_optimizer, device)    
        
        # Save model parameters
        model_path = "/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_epoch_" + str(t+1)
        torch.save(ft_resnet50.state_dict(), model_path)

        with open("/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_last_epoch.txt", 'w') as f:
            f.write(str(t+1))

        # Check performance on the validation set
        v_loss, v_accuracy = validation(validation_data_loader, ft_resnet50, loss_fn, device)
        
        # Save training and validation set performance
        train_loss.loc[t, 'ft_resnet'] = t_loss
        validation_loss.loc[t, 'ft_resnet'] = v_loss
        train_accuracy.loc[t, 'ft_resnet'] = t_accuracy
        validation_accuracy.loc[t, 'ft_resnet'] = v_accuracy
        train_loss.to_csv("/content/gdrive/My Drive/colab/results/train_loss.csv")
        validation_loss.to_csv("/content/gdrive/My Drive/colab/results/validation_loss.csv")
        train_accuracy.to_csv("/content/gdrive/My Drive/colab/results/train_accuracy.csv")
        validation_accuracy.to_csv("/content/gdrive/My Drive/colab/results/validation_accuracy.csv")

        # Save state dict of best performing epoch
        if v_loss < best_v_loss:
          best_v_loss = v_loss
          model_path = "/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_best_epoch"
          torch.save(ft_resnet50.state_dict(), model_path)
          with open("/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_best_val_loss.txt", 'w') as f:
            f.write(str(best_v_loss))

def fit_pt_resnet(train_data_loader, pt_resnet50, loss_fn, pt_resnet50_optimizer, device, epochs, restart, validation_data_loader, train_loss, validation_loss, train_accuracy, validation_accuracy):
    # Load the model from the latest trained epoch  
    with open("/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_last_epoch.txt") as f:
        start_epoch = int(f.readlines()[0])
    
    if start_epoch > 0:
        load_model_path = "/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_epoch_" + str(start_epoch)
        pt_resnet50.load_state_dict(torch.load(load_model_path))

    # Keep track of the best performance on the validation set
    with open("/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_best_val_loss.txt") as f:
      best_v_loss = float(f.readlines()[0])  

    # Train untill the required epoch
    print("Retrain pretrained ResNet50")
    for t in range(start_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # Decrease the learning rate and weight decay after training for 5 and 10 epochs
        if t >=5:
          pt_resnet50_optimizer = torch.optim.Adam(pt_resnet50.parameters(), lr = 1e-4, weight_decay = 1e-5)

        if t >=10:
          pt_resnet50_optimizer = torch.optim.Adam(pt_resnet50.parameters(), lr = 1e-5, weight_decay = 1e-6)
        
        # Train a single epoch
        t_loss, t_accuracy = train_epoch(train_data_loader, pt_resnet50, loss_fn, pt_resnet50_optimizer, device)    
        
        # Save model parameters
        model_path = "/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_epoch_" + str(t+1)
        torch.save(pt_resnet50.state_dict(), model_path)

        with open("/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_last_epoch.txt", 'w') as f:
            f.write(str(t+1))

        # Check performance on the validation set
        v_loss, v_accuracy = validation(validation_data_loader, pt_resnet50, loss_fn, device)
        
        # Save training and validation set performance
        train_loss.loc[t, 'pt_resnet'] = t_loss
        validation_loss.loc[t, 'pt_resnet'] = v_loss
        train_accuracy.loc[t, 'pt_resnet'] = t_accuracy
        validation_accuracy.loc[t, 'pt_resnet'] = v_accuracy
        train_loss.to_csv("/content/gdrive/My Drive/colab/results/train_loss.csv")
        validation_loss.to_csv("/content/gdrive/My Drive/colab/results/validation_loss.csv")
        train_accuracy.to_csv("/content/gdrive/My Drive/colab/results/train_accuracy.csv")
        validation_accuracy.to_csv("/content/gdrive/My Drive/colab/results/validation_accuracy.csv")

        # Save state dict of best performing epoch
        if v_loss < best_v_loss:
          best_v_loss = v_loss
          model_path = "/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_best_epoch"
          torch.save(pt_resnet50.state_dict(), model_path)
          with open("/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_best_val_loss.txt", 'w') as f:
            f.write(str(best_v_loss))

def fit_resnet(train_data_loader, resnet50, loss_fn, resnet50_optimizer, device, epochs, restart, validation_data_loader, train_loss, validation_loss, train_accuracy, validation_accuracy):
    # Load the model from the latest trained epoch  
    with open("/content/gdrive/My Drive/colab/models/pcam_resnet50_last_epoch.txt") as f:
        start_epoch = int(f.readlines()[0])
    
    if start_epoch > 0:
        load_model_path = "/content/gdrive/My Drive/colab/models/pcam_resnet50_epoch_" + str(start_epoch)
        resnet50.load_state_dict(torch.load(load_model_path))

    # Keep track of the best performance on the validation set
    with open("/content/gdrive/My Drive/colab/models/pcam_resnet50_best_val_loss.txt") as f:
      best_v_loss = float(f.readlines()[0])  

    # Train untill the required epoch
    print("Train ResNet50")
    for t in range(start_epoch, epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # Decrease the learning rate and weight decay after training for 5 and 10 epochs
        if t >=5:
          resnet50_optimizer = torch.optim.Adam(resnet50.parameters(), lr = 1e-4, weight_decay = 1e-5)

        if t >=10:
          resnet50_optimizer = torch.optim.Adam(resnet50.parameters(), lr = 1e-5, weight_decay = 1e-6)
        
        # Train a single epoch
        t_loss, t_accuracy = train_epoch(train_data_loader, resnet50, loss_fn, resnet50_optimizer, device)    
        
        # Save model parameters
        model_path = "/content/gdrive/My Drive/colab/models/pcam_resnet50_epoch_" + str(t+1)
        torch.save(resnet50.state_dict(), model_path)

        with open("/content/gdrive/My Drive/colab/models/pcam_resnet50_last_epoch.txt", 'w') as f:
            f.write(str(t+1))

        # Check performance on the validation set
        v_loss, v_accuracy = validation(validation_data_loader, resnet50, loss_fn, device)
        
        # Save training and validation set performance
        train_loss.loc[t, 'resnet'] = t_loss
        validation_loss.loc[t, 'resnet'] = v_loss
        train_accuracy.loc[t, 'resnet'] = t_accuracy
        validation_accuracy.loc[t, 'resnet'] = v_accuracy
        train_loss.to_csv("/content/gdrive/My Drive/colab/results/train_loss.csv")
        validation_loss.to_csv("/content/gdrive/My Drive/colab/results/validation_loss.csv")
        train_accuracy.to_csv("/content/gdrive/My Drive/colab/results/train_accuracy.csv")
        validation_accuracy.to_csv("/content/gdrive/My Drive/colab/results/validation_accuracy.csv")

        # Save state dict of best performing epoch
        if v_loss < best_v_loss:
          best_v_loss = v_loss
          model_path = "/content/gdrive/My Drive/colab/models/pcam_resnet50_best_epoch"
          torch.save(resnet50.state_dict(), model_path)
          with open("/content/gdrive/My Drive/colab/models/pcam_resnet50_best_val_loss.txt", 'w') as f:
            f.write(str(best_v_loss))

def train_epoch(dataloader, train_model, loss_fn, train_optimizer, device):
    size = len(dataloader.dataset)
    train_optimizer = torch.optim.Adam(train_model.parameters(), lr=1e-3)
    train_model.train()
    
    total_loss, total_correct = 0, 0    
    for batch, (X, y) in enumerate(dataloader):
        # Make sure the tensors are set to be processed by the correct device
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = train_model(X)
        loss = loss_fn(pred, y.unsqueeze(1).float())
        
        # Backpropagation
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        

        # Performance tracking
        m = torch.sigmoid(pred)
        bin_pred = torch.round(m).transpose(0,1)
        total_correct += (bin_pred == y).type(torch.float).sum().item()
        total_loss += loss
        
        

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            tmp_num=0
            for param in train_model.parameters():
              print(param.data.sum())
              try:
                print(param.grad.data.sum())
              except:
                print("Freezed gradients")
              tmp_num+=1
              if tmp_num > 5:
                print("")
                break

        
    total_loss/=len(dataloader)
    total_correct/=size

    return total_loss.item(), (total_correct*100)

class Experiment():
    def __init__(self, epochs, restart):
        self.num_epochs = epochs
        self.restart = restart
        if self.restart:
            self.train_loss = pd.DataFrame(index=range(self.num_epochs), columns=['convnet', 'ft_resnet', 'pt_resnet', 'resnet'])
            self.validation_loss = pd.DataFrame(index=range(self.num_epochs), columns=['convnet', 'ft_resnet', 'pt_resnet', 'resnet'])
            self.train_accuracy = pd.DataFrame(index=range(self.num_epochs), columns=['convnet', 'ft_resnet', 'pt_resnet', 'resnet'])
            self.validation_accuracy = pd.DataFrame(index=range(self.num_epochs), columns=['convnet', 'ft_resnet', 'pt_resnet', 'resnet'])
        else:
            self.train_loss = pd.read_csv("/content/gdrive/My Drive/colab/results/train_loss.csv")
            self.validation_loss = pd.read_csv("/content/gdrive/My Drive/colab/results/validation_loss.csv")
            self.train_accuracy = pd.read_csv("/content/gdrive/My Drive/colab/results/train_accuracy.csv")
            self.validation_accuracy = pd.read_csv("/content/gdrive/My Drive/colab/results/validation_accuracy.csv")

    def run(self, train_data_loader, conv_model, ft_resnet50, pt_resnet50, resnet50, loss_fn, conv_optimizer, ft_resnet50_optimizer, pt_resnet50_optimizer, resnet50_optimizer, device, validation_data_loader):
        if self.restart:
          # Save starting epochs for interrupted training
          with open("/content/gdrive/My Drive/colab/models/pcam_conv_last_epoch.txt", 'w') as f:
              f.write(str(0))
          with open("/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_last_epoch.txt", 'w') as f:
              f.write(str(0))
          with open("/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_last_epoch.txt", 'w') as f:
              f.write(str(0))
          with open("/content/gdrive/My Drive/colab/models/pcam_resnet50_last_epoch.txt", 'w') as f:
              f.write(str(0))
          # Save the best performing validation loss for interrupted training
          with open("/content/gdrive/My Drive/colab/models/pcam_conv_best_val_loss.txt", 'w') as f:
              f.write(str(100))
          with open("/content/gdrive/My Drive/colab/models/pcam_ft_resnet50_best_val_loss.txt", 'w') as f:
              f.write(str(100))
          with open("/content/gdrive/My Drive/colab/models/pcam_pt_resnet50_best_val_loss.txt", 'w') as f:
              f.write(str(100))
          with open("/content/gdrive/My Drive/colab/models/pcam_resnet50_best_val_loss.txt", 'w') as f:
              f.write(str(100))
        fit_convnet(train_data_loader, conv_model, loss_fn, conv_optimizer, device, self.num_epochs, self.restart, validation_data_loader, self.train_loss, self.validation_loss, self.train_accuracy, self.validation_accuracy)
        fit_ft_resnet(train_data_loader, ft_resnet50, loss_fn, ft_resnet50_optimizer, device, self.num_epochs, self.restart, validation_data_loader, self.train_loss, self.validation_loss, self.train_accuracy, self.validation_accuracy)
        fit_pt_resnet(train_data_loader, pt_resnet50, loss_fn, pt_resnet50_optimizer, device, self.num_epochs, self.restart, validation_data_loader, self.train_loss, self.validation_loss, self.train_accuracy, self.validation_accuracy)
        fit_resnet(train_data_loader, resnet50, loss_fn, resnet50_optimizer, device, self.num_epochs, self.restart, validation_data_loader, self.train_loss, self.validation_loss, self.train_accuracy, self.validation_accuracy)
