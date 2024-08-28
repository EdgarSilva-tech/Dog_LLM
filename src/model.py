import torch
from torch import nn
from data_loader import load_data
from tqdm.auto import tqdm
import torchvision
import os

train_dataloader, val_dataloader, class_names = load_data()

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    y_pred_classes = torch.argmax(y_pred, dim=1)
    correct = torch.eq(y_true, y_pred_classes).sum().item()
    acc = (correct / len(y_true)) * 100
    return acc


def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               epochs: int):

    torch.manual_seed(42)

    for epoch in tqdm(range(epochs)):

        model.train()
        train_loss, correct_preds, total_samples = 0, 0, 0

        for batch, (X, y) in enumerate(data_loader):

            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            train_loss += loss.item()

            y_pred_classes = torch.argmax(y_pred, dim=1)
            correct_batch = torch.eq(y, y_pred_classes).sum().item()
            correct_preds += correct_batch
            total_samples += len(y)

            print(f"Batch {batch}: correct {correct_batch}, total {len(y)}, cumulative correct {correct_preds}, cumulative total {total_samples}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(data_loader)
        train_acc = (correct_preds / total_samples) * 100

        print(f"Train loss: {train_loss:.5f} | Train accuracy {train_acc:.2f}%")

weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights from pretraining on ImageNet

auto_transforms = weights.transforms()

model = torchvision.models.efficientnet_b0(weights=weights)

for param in model.features.parameters():
    param.requires_grad = False

output_shape = len(os.listdir('data/train'))

# Recreate the classifier layer and seed it to the target device
model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True))

model.load_state_dict(torch.load('artifacts/model_state.pth', map_location=torch.device("cpu")))

def predict(image: str):

    #img = Image.open(image)

    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = auto_transforms(image).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image)

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    label_pred = class_names[target_image_pred_label]

    return label_pred

if __name__ ==  '__main__':
    train_dataloader, val_dataloader, class_names = load_data()
    # train_features_batch, train_labels_batch = next(iter(train_dataloader))
    # print(train_features_batch.shape, train_labels_batch.shape)


    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), 
                            lr=0.05)

    train_step(model, train_dataloader, loss_fn, optimizer, 10)

    torch.save(model.state_dict(), 'artifacts/model_state.pth')
