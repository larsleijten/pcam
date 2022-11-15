def validation(dataloader, val_model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    val_model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = val_model(X)
            m = torch.sigmoid(pred)
            bin_pred = torch.round(m).transpose(0,1)
            test_loss += loss_fn(pred, y.unsqueeze(1).float()).item()
            correct += (bin_pred == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, (100*correct)

def tta_test(test_data_loader, test_model, loss_fn, device):
  size = len(test_data_loader.dataset)
  num_batches = len(test_data_loader)
  
  # d4-Transform: Accuracy from 88.1 to 88.4
  tta_model = tta.ClassificationTTAWrapper(test_model, tta.aliases.flip_transform())
  
  total_predictions = []
  
  tta_model.eval()
  test_loss, correct, auroc = 0, 0, 0
  auroc_func = AUROC(pos_label=1)
  auroc_func = BinaryAUROC()
  with torch.no_grad():
      for X, y in test_data_loader:
          X, y = X.to(device), y.to(device)
          pred = tta_model(X)
          m = torch.sigmoid(pred)
          total_predictions.append(torch.squeeze(m).tolist())
          bin_pred = torch.round(m).transpose(0,1)
          auroc += auroc_func(torch.squeeze(torch.transpose(m, 0, 1)), y).item()
          test_loss += loss_fn(pred, y.unsqueeze(1).float()).item()
          correct += (bin_pred == y).type(torch.float).sum().item()     
  test_loss /= num_batches
  correct /= size
  auroc /= num_batches
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}, AUROC: {auroc:>8f} \n")
  total_predictions = [item for sublist in total_predictions for item in sublist]
  return total_predictions