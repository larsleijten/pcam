# Generate data loaders with incorporated data augmentation
def load_data():
    
    # Mounts my Google Drive to Colab
    from google.colab import drive
    drive.mount('/content/gdrive')

    ### Copy all the data sets from Drive to Colab

    # Train

    colab_link = "/content/data/pcam/"
    gdrive_link = "/content/gdrive/My Drive/colab/datasets/pcam/camelyonpatch_level_2_split_train_x.h5.gz"
    shutil.copy(gdrive_link, colab_link)
    colab_link = "/content/data/pcam/"
    gdrive_link = "/content/gdrive/My Drive/colab/datasets/pcam/camelyonpatch_level_2_split_train_y.h5.gz"
    shutil.copy(gdrive_link, colab_link)

    # Validation

    colab_link = "/content/data/pcam/"
    gdrive_link = "/content/gdrive/My Drive/colab/datasets/pcam/camelyonpatch_level_2_split_test_x.h5.gz"
    shutil.copy(gdrive_link, colab_link)
    colab_link = "/content/data/pcam/"
    gdrive_link = "/content/gdrive/My Drive/colab/datasets/pcam/camelyonpatch_level_2_split_test_y.h5.gz"
    shutil.copy(gdrive_link, colab_link)

    # Test

    colab_link = "/content/data/pcam/"
    gdrive_link = "/content/gdrive/My Drive/colab/datasets/pcam/camelyonpatch_level_2_split_valid_x.h5.gz"
    shutil.copy(gdrive_link, colab_link)
    colab_link = "/content/data/pcam/"
    gdrive_link = "/content/gdrive/My Drive/colab/datasets/pcam/camelyonpatch_level_2_split_valid_y.h5.gz"
    shutil.copy(gdrive_link, colab_link)

    # Import Data from PCAM dataset
    train_data = datasets.PCAM(
        split = "train",
        root="data",
        download=True,
        # Define data augmentations (are re-augmented each epoch)
        transform=transforms.Compose([transforms.RandomRotation(degrees=(0, 180)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomCrop(80, padding=8),
                                    transforms.RandomInvert(p=0.3),
                                    transforms.ColorJitter(brightness=0.5, hue=.3),
                                    transforms.RandomGrayscale(p=0.3),
                                    ToTensor()])
    
    )

    # After an MD5 Checksum error, I had to rename the data files, the validation and test were switched
    validation_data = datasets.PCAM(
        split = "test",
        root="data",
        download=True,
        transform=transforms.Compose([ToTensor()])
    )

    test_data = datasets.PCAM(
        split = "val",
        root="data",
        download=True,
        transform=transforms.Compose([ToTensor()])
    )

    # Create Data loaders
    batch_size = 128
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle = True)
    validation_data_loader = DataLoader(validation_data, batch_size=batch_size*2)
    test_data_loader = DataLoader(test_data, batch_size=batch_size*2)

    return train_data_loader, validation_data_loader, test_data_loader
