class ConvNet(nn.Module):
  def __init__(self):
    super().__init__()
    # Build a small Convolutional Neural Network
    self.conv = nn.Sequential(
        nn.Conv2d(3,32,kernel_size=3,padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(4,4),

        nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(4,4),

        nn.Flatten(),
        nn.Linear(2304,1024),
        nn.LeakyReLU(),
        nn.Linear(1024,512),
        nn.LeakyReLU(),
        nn.Linear(512,1)
    )
  
  def forward(self, x):
    logits = self.conv(x)
    return logits