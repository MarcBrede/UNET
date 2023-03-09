import torch
from torch import nn
import numpy as np 
from matplotlib import pyplot as plt
from torch.optim import Adam
import torch.nn.functional as F

from simple_deep_learning.mnist_extended.semantic_segmentation import create_semantic_segmentation_dataset
from simple_deep_learning.mnist_extended.semantic_segmentation import display_grayscale_array, plot_class_masks, display_segmented_image
from custom.dataset import SegmentationDataloader
from custom.model import SimpleNet, UNET_SMALL
from custom.unet import UNET
from custom.loss import WeightedBCE

NUM_CLASSES = 2
NUM_EPOCHS = 20
LR = 0.001
BATCH_SIZE = 32
NAME = "UNET_2_classes"

np.random.seed(1)
train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=1000,
                                                                        num_test_samples=200,
                                                                        image_shape=(60, 60),
                                                                        max_num_digits_per_image=4,
                                                                        num_classes=NUM_CLASSES)

model = UNET(in_channels=1, num_classes=NUM_CLASSES)

# pytorch expexts the channels to be the first dimension
train_x = torch.moveaxis(torch.tensor(train_x, dtype=torch.float32), 3, 1)
train_y = torch.moveaxis(torch.tensor(train_y, dtype=torch.float32), 3, 1)
test_x = torch.moveaxis(torch.tensor(test_x, dtype=torch.float32), 3, 1)
test_y = torch.moveaxis(torch.tensor(test_y, dtype=torch.float32), 3, 1)

# convert to boolean labels and make them exclusive
new_train_y = torch.zeros_like(train_y).scatter(1, train_y.argmax(dim=1, keepdim=True), value=1)
new_train_y[train_y < 0.5] = 0
train_y = new_train_y

new_test_y = torch.zeros_like(test_y).scatter(1, test_y.argmax(dim=1, keepdim=True), value=1)
new_test_y[test_y < 0.5] = 0
test_y = new_test_y

optimizer = Adam(
        model.parameters(), 
        lr = LR,
    )

class_dis = torch.cat((torch.sum(train_y == 0)[None], torch.count_nonzero(train_y, dim=[0,2,3])))
loss_fn = nn.BCELoss()
# loss_fn = WeightedBCE(weights=1 - (class_dis/class_dis.sum()))
train_loader = SegmentationDataloader(train_x, train_y, batch_size=BATCH_SIZE)
eval_loader = SegmentationDataloader(test_x, test_y, batch_size=BATCH_SIZE)

def train():
    model.train()
    loss_arr = np.array([])
    optimizer.zero_grad()
    for input in train_loader:
        x, y = input
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        loss_arr = np.append(loss_arr, loss.detach().numpy().item())
    print(f"Training loss: {loss.mean()}")
    return loss.detach().numpy().mean()

def eval():
    model.eval()
    loss_arr = np.array([])
    with torch.no_grad():
        for input in eval_loader:
            x, y = input
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss_arr = np.append(loss_arr, loss.detach().numpy().item())
    print(f"Validation loss: {loss.mean()}")
    return loss.detach().numpy().mean()

# LOGS = {
#     "train_loss": [],
#     "eval_loss": []
# }

# best_eval_loss = float("inf")
# for epoch in range(NUM_EPOCHS):
#     print(f"Epoch {epoch}")
#     train_loss = train()
#     eval_loss = eval()
#     LOGS["train_loss"].append(train_loss)
#     LOGS["eval_loss"].append(eval_loss)
#     if eval_loss < best_eval_loss:
#         best_eval_loss = eval_loss
#         print(f"New best model found: {best_eval_loss}")
#         torch.save(model.state_dict(), "./{}.pth".format(NAME))

# plt.plot(LOGS["train_loss"], label="train")
# plt.plot(LOGS["eval_loss"], label="eval")
# plt.legend()
# plt.show()

model.load_state_dict(torch.load("./{}.pth".format(NAME)))
model.eval()
for i in range(10):
    display_segmented_image(torch.moveaxis(test_y, 1, 3)[i])
    display_segmented_image(torch.moveaxis(model(test_x[i: i+1]), 1, 3)[0], threshold=0.1)

model.load_state_dict(torch.load("./{}.pth".format(NAME)))


