# Calculate the loss and accuracy over a dataset (often validation set)
def validation(dataloader, val_model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_model.eval()
    test_loss, correct = 0, 0
    # Do not keep track of gradients
    with torch.no_grad():
        # Loop over the batches in the dataloader
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            # Model predictions
            pred = val_model(X)
            # Activate model predictions with a sigmoid
            m = torch.sigmoid(pred)
            # Make a binary prediction at the threshold of 0.5
            bin_pred = torch.round(m).transpose(0,1)
            # Keep track of loss and accuracy
            test_loss += loss_fn(pred, y.unsqueeze(1).float()).item()
            correct += (bin_pred == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, (100*correct)

# Apply Test Time Augmentation and calculate the loss, accuracy and AUROC over a dataset
def tta_test(test_data_loader, test_model, loss_fn, device):
    size = len(test_data_loader.dataset)
    num_batches = len(test_data_loader)
    
    # Wrap the model in a TTA model which flip-augments the images
    tta_model = tta.ClassificationTTAWrapper(test_model, tta.aliases.flip_transform())
    
    total_predictions = []
    
    tta_model.eval()
    test_loss, correct, auroc = 0, 0, 0
    auroc_func = BinaryAUROC()
    # Do not keep track of gradients
    with torch.no_grad():
        # Loop over the batches in the dataloader
        for X, y in test_data_loader:
            X, y = X.to(device), y.to(device)
            # Model predictions
            pred = tta_model(X)
            # Activate model predictions with a sigmoid
            m = torch.sigmoid(pred)           
            # Make a binary prediction at the threshold of 0.5
            bin_pred = torch.round(m).transpose(0,1)
            # Keep track of AUROC, loss and accuracy
            auroc += auroc_func(torch.squeeze(torch.transpose(m, 0, 1)), y).item()
            test_loss += loss_fn(pred, y.unsqueeze(1).float()).item()
            correct += (bin_pred == y).type(torch.float).sum().item()
            total_predictions.append(torch.squeeze(m).tolist())     
    test_loss /= num_batches
    correct /= size
    auroc /= num_batches
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, AUROC: {auroc:>8f} \n")
    # Put all the model predictions in a single list
    total_predictions = [item for sublist in total_predictions for item in sublist]
    return total_predictions