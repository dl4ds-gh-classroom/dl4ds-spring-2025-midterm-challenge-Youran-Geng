import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm  # For progress bars
import wandb
import json

# safely download the pretrained model weights
import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())
from torchvision.models import ResNet18_Weights

################################################################################
# Model Definition (Simple Example - You need to complete)
# For Part 1, you need to manually define a network.
# For Part 2 you have the option of using a predefined network and
# for Part 3 you have the option of using a predefined, pretrained network to
# finetune.
################################################################################
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # TODO - define the layers of the network you will use
        # 2 convolutional layers, 2 fully connected layers
        # the input is (R, G, B) so the first conv layer should have in_channels=3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, 100)  # 100 classes in CIFAR-100
    
    def forward(self, x):
        # TODO - define the forward pass of the network you will use
        # use ReLU activation
        # pooling after the first and second conv layers
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(x)
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

################################################################################
# Define a one epoch training function
################################################################################
def train(epoch, model, trainloader, optimizer, criterion, CONFIG):
    """Train one epoch, e.g. all batches of one epoch."""
    device = CONFIG["device"]
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    # put the trainloader iterator in a tqdm so it can printprogress
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", leave=False)

    # iterate through all batches of one epoch
    for i, (inputs, labels) in enumerate(progress_bar):

        # move inputs and labels to the target device
        inputs, labels = inputs.to(device), labels.to(device)

        ### TODO - Your code here
        optimizer.zero_grad()  # clear the parameter gradients
        outputs = model(inputs) # forward pass
        loss = criterion(outputs, labels)
        loss.backward()  # backpropagation
        optimizer.step()  # update the weights

        # add losses and predictions
        running_loss += loss.item()   ### TODO
        _, predicted = outputs.max(1)    ### TODO

        # calculate accuracy
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({"loss": running_loss / (i + 1), "acc": 100. * correct / total})

    train_loss = running_loss / len(trainloader)
    train_acc = 100. * correct / total
    return train_loss, train_acc


################################################################################
# Define a validation function
################################################################################
def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval() # Set to evaluation
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad(): # No need to track gradients
        
        # Put the valloader iterator in tqdm to print progress
        progress_bar = tqdm(valloader, desc="[Validate]", leave=False)

        # Iterate throught the validation set
        for i, (inputs, labels) in enumerate(progress_bar):
            
            # move inputs and labels to the target device
            inputs, labels = inputs.to(device), labels.to(device)

            # similar to training but no backpropagation/optimizer
            outputs = model(inputs) ### TODO -- inference
            loss = criterion(outputs, labels)    ### TODO -- loss calculation

            running_loss += loss.item()  ### SOLUTION -- add loss from this sample
            _, predicted = outputs.max(1)   ### SOLUTION -- predict the class

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            progress_bar.set_postfix({"loss": running_loss / (i+1), "acc": 100. * correct / total})

    val_loss = running_loss/len(valloader)
    val_acc = 100. * correct / total
    return val_loss, val_acc


def main():

    ############################################################################
    #    Configuration Dictionary (Modify as needed)
    ############################################################################
    # It's convenient to put all the configuration in a dictionary so that we have
    # one place to change the configuration.
    # It's also convenient to pass to our experiment tracking tool.


    CONFIG = {
        "model": SimpleCNN,   # Change name when using a different model; see model instantiation below
        "batch_size": 256, # run batch size finder to find optimal batch size
        "learning_rate": 0.1,
        "epochs": 25,  # Train for longer in a real scenario
        "num_workers": 6, # Adjust based on your system
        "device": "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu",
        "data_dir": "./data",  # Make sure this directory exists
        "ood_dir": "./data/ood-test",
        "wandb_project": "sp25-ds542-challenge",
        "seed": 42,
    }

    import pprint
    print("\nCONFIG Dictionary:")
    pprint.pprint(CONFIG)

    ############################################################################
    #      Data Transformation (Example - You might want to modify) 
    ############################################################################

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Example normalization
    ])

    ###############
    # TODO Add validation and test transforms - NO augmentation for validation/test
    ###############

    # Validation and test transforms (NO augmentation)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # same normalization as training
    ])   ### TODO -- BEGIN SOLUTION

    ############################################################################
    #  PART 2: Using ResNet18 pretrained. Need to upscale the images to 224x224
    # comment the following when using SimpleCNN
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # Example normalization
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # same normalization as training
    ])
    ############################################################################

    ############################################################################
    #       Data Loading
    ############################################################################

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)

    # Split train into train and validation (80/20 split)
    train_size = int(0.8 * len(trainset))   ### TODO -- Calculate training set size
    val_size = len(trainset) - train_size     ### TODO -- Calculate validation set size
    print(f"Train size: {train_size}, Validation size: {val_size}")
    trainset, valset = random_split(trainset, [train_size, val_size])  ### TODO -- split into training and validation sets

    ### TODO -- define loaders and test set
    trainloader = DataLoader(trainset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"])
    valloader = DataLoader(valset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])

    # ... (Create validation and test loaders)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"])
    
    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    model = CONFIG["model"]()   # instantiate your model ### TODO
    model = model.to(CONFIG["device"])   # move it to target device

    ############################################################################
    # PART 2: Using ResNet18 pretrained
    # comment following when using SimpleCNN
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # Load the pretrained ResNet18 model
    model.fc = nn.Linear(model.fc.in_features, 100) # Last layer need to be 100
    model = model.to(CONFIG["device"])
    ############################################################################
    
    ############################################################################
    # PART 3: Using ResNet18 pretrained and finetuning
    # freeze the model parameters except the last layer (fc) for finetuning
    if isinstance(model, models.ResNet):
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze the last layer (fc) for training
        for param in model.fc.parameters():
            param.requires_grad = True
    ############################################################################

    print("\nModel summary:")
    print(f"{model}\n")

    # The following code you can run once to find the batch size that gives you the fastest throughput.
    # You only have to do this once for each machine you use, then you can just
    # set it in CONFIG.
    SEARCH_BATCH_SIZES = False
    if SEARCH_BATCH_SIZES:
        from utils import find_optimal_batch_size
        print("Finding optimal batch size...")
        optimal_batch_size = find_optimal_batch_size(model, trainset, CONFIG["device"], CONFIG["num_workers"])
        CONFIG["batch_size"] = optimal_batch_size
        print(f"Using batch size: {CONFIG['batch_size']}")
    

    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    criterion = nn.CrossEntropyLoss()   ### TODO -- define loss criterion
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])   ### TODO -- define optimizer
    # PART 3: If using ResNet18 and finetuning
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=CONFIG['learning_rate'], momentum=0.9)   ### TODO -- define optimizer
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # Add a scheduler   ### TODO -- you can optionally add a LR scheduler


    # Initialize wandb
    wandb.init(project="-sp25-ds542-challenge", config=CONFIG)
    wandb.watch(model)  # watch the model gradients

    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ############################################################################
    best_val_acc = 0.0

    # for epoch in range(CONFIG["epochs"]):
    #     train_loss, train_acc = train(epoch, model, trainloader, optimizer, criterion, CONFIG)
    #     val_loss, val_acc = validate(model, valloader, criterion, CONFIG["device"])
    #     scheduler.step()

    #     # log to WandB
    #     wandb.log({
    #         "epoch": epoch + 1,
    #         "train_loss": train_loss,
    #         "train_acc": train_acc,
    #         "val_loss": val_loss,
    #         "val_acc": val_acc,
    #         "lr": optimizer.param_groups[0]["lr"] # Log learning rate
    #     })

    #     # Save the best model (based on validation accuracy)
    #     if val_acc > best_val_acc:
    #         best_val_acc = val_acc
    #         torch.save(model.state_dict(), "best_model.pth")
    #         wandb.save("best_model.pth") # Save to wandb as well
    #     # Save the model every 5 epochs
    #     if (epoch + 1) % 5 == 0:
    #         torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
    #         wandb.save(f"model_epoch_{epoch+1}.pth")
    #         print(f"Model saved at epoch {epoch+1}")

    wandb.finish()

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################
    import eval_cifar100
    import eval_ood

    # use best model
    model.load_state_dict(torch.load("best_model.pth", map_location=CONFIG["device"]))
    model = model.to(CONFIG["device"])
    model.eval()  # Set to evaluation mode

    
    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = eval_cifar100.evaluate_cifar100_test(model, testloader, CONFIG["device"])
    print(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = eval_ood.evaluate_ood_test(model, CONFIG)

    # --- Create Submission File (OOD) ---
    submission_df_ood = eval_ood.create_ood_df(all_predictions)
    submission_df_ood.to_csv("submission_ood.csv", index=False)
    print("submission_ood.csv created successfully.")

if __name__ == '__main__':
    main()
