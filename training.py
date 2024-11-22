import torch
import torch.nn as nn
import monai
from monai.losses import DiceLoss

from utils import train_test_split, Lambda, printLambda

def train_model(modelFunc, model, BATCH_SIZE: int = 10, LR: float = 1e-3, EPOCHS: int = 20):
    """
    Our main training loop.

    :param modelFunc: A function that takes in a tensor
      of shape (B, W, H, numSlices, 4), and outputs a tensor
      of shape (B, W, H, numSlices, 1) that contains the segmentation
      prediction
    :param model: The model to train, as an nn.Module
    :param BATCH_SIZE: The batch size to use
    :param LR: The learning rate to use
    :param EPOCHS: The number of epochs to train for
    """

    # First, build up our data utils
    trainDataset, testDataset = train_test_split()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    trLoad = monai.data.DataLoader(trainDataset, batch_size=BATCH_SIZE)
    tsLoad = monai.data.DataLoader(testDataset, batch_size=BATCH_SIZE)

    # Get some examples we can use for reshaping
    EX_IN, EX_OUT = trainDataset[0]

    optim = torch.optim.Adam(model.parameters(), lr=LR)

    diceLoss = DiceLoss(reduction="mean", sigmoid=True)

    # Now, train.
    for epochIdx in range(EPOCHS):
        model.train()
        for batchIdx, batch in enumerate(iter(trLoad)):
            inp, out = batch

            inp = inp.to(device, non_blocking=True)
            out = out.to(device, non_blocking=True)

            modelPred = modelFunc(inp)

            loss = diceLoss(modelPred, out)
            print(f"Train Loss: {loss.item()}")

            optim.zero_grad()
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            for batchIdx, batch in enumerate(iter(tsLoad)):
                inp, out = batch

                inp = inp.to(device, non_blocking=True)
                out = out.to(device, non_blocking=True)

                modelPred = modelFunc(inp)
                val_loss = diceLoss(modelPred, out)
                print(f"Validation Loss: {val_loss.item()}")

    return model
