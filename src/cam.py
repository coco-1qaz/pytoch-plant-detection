import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
import torchvision.transforms as transforms

from tqdm.auto import tqdm
from model import build_model
from torch.utils.data import DataLoader
from torchvision import datasets
from torch import topk

# Constants and other configurations.
TEST_DIR = os.path.join('..', 'input', 'Test', 'Test')
BATCH_SIZE = 1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
IMAGE_RESIZE = 224
NUM_WORKERS = 4
CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']

# Validation transforms
def get_test_transform(image_size):
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
    ])
    return test_transform

def get_datasets(image_size):
    """
    Function to prepare the Datasets.
    Returns the test dataset.
    """
    dataset_test = datasets.ImageFolder(
        TEST_DIR, 
        transform=(get_test_transform(image_size))
    )
    return dataset_test

def get_data_loader(dataset_test):
    """
    Prepares the training and validation data loaders.
    :param dataset_test: The test dataset.

    Returns the training and validation data loaders.
    """
    test_loader = DataLoader(
        dataset_test, batch_size=BATCH_SIZE, 
        shuffle=False, num_workers=NUM_WORKERS
    )
    return test_loader


def denormalize(
    x, 
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
):
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(x, 0, 1)

# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def show_cam(
    CAMs, 
    image, 
    labels, 
    output_class, 
    save_name
):
    for i, cam in enumerate(CAMs):
        image = denormalize(image).cpu()
        image = image.squeeze(0).permute((1, 2, 0)).numpy()
        image = np.ascontiguousarray(image, dtype=np.float32)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        orig_image = image.copy()
        gt = labels.cpu().numpy()
        cv2.putText(
            image, f"GT: {CLASS_NAMES[int(gt)]}", 
            (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, (0, 255, 0), 2, cv2.LINE_AA
        )
        if output_class == gt:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.putText(
            image, f"Pred: {CLASS_NAMES[int(output_class)]}", 
            (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, color, 2, cv2.LINE_AA
        )
        height, width, _ = image.shape
        heatmap = cv2.applyColorMap(cv2.resize(cam, (width, height)), cv2.COLORMAP_JET)
        result = heatmap/255. * 0.5 + image * 0.5
        # cv2.imshow('CAM', result)
        # cv2.waitKey(0)
        result = np.array(result, dtype=np.float32)
        image_concat = cv2.hconcat([orig_image, result])
        cv2.imwrite(save_name, image_concat*255.)
        cv2.destroyAllWindows()

def test(model, testloader, DEVICE
):
    """
    Function to test the trained model on the test dataset.

    :param model: The trained model.
    :param testloader: The test data loader.
    :param DEVICE: The computation device.

    Returns:
        predictions_list: List containing all the predicted class numbers.
        ground_truth_list: List containing all the ground truth class numbers.
        acc: The test accuracy.
    """
    model.eval()
    print('Testing model')
    predictions_list = []
    ground_truth_list = []
    test_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):

            ######### CAM CODE #########
            # Hook the feature extractor.
            features_blobs = []
            def hook_feature(module, input, output):
                features_blobs.append(output.data.cpu().numpy())
            model._modules.get('layer4').register_forward_hook(hook_feature)
            # Get the softmax weight
            params = list(model.parameters())
            # weight_softmax = np.squeeze(params[-4].data.cpu().numpy())

            weight_softmax = params[-2]
            weight_softmax = F.relu(weight_softmax)
            weight_softmax = np.squeeze(weight_softmax.data.cpu().numpy())
            ######### CAM CODE #########

            counter += 1
            image, labels = data
            image = image.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass.
            outputs = model(image)
            # Softmax probabilities.
            predictions = F.softmax(outputs, dim=1).cpu().numpy()
            # Predicted class number.
            output_class = np.argmax(predictions)
            # Append the GT and predictions to the respective lists.
            predictions_list.append(output_class)
            ground_truth_list.append(labels.cpu().numpy())
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            test_running_correct += (preds == labels).sum().item()

            # For CAM.
            # Get the softmax probabilities.
            probs = F.softmax(outputs, dim=1).data.squeeze()
            # Get the class indices of top k probabilities.
            class_idx = topk(probs, 1)[1].int()
            # Generate class activation mapping for the top1 prediction.
            CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
            # file name to save the resulting CAM image with
            save_name = os.path.join(
                '..',
                'outputs',
                'cam_results',
                str(counter)+'.png'
            )
            # Show and save the results.
            show_cam(
                CAMs, 
                image, 
                labels, 
                output_class, 
                save_name
            )

    acc = 100. * (test_running_correct / len(testloader.dataset))
    return predictions_list, ground_truth_list, acc

if __name__ == '__main__':
    os.makedirs(os.path.join('..', 'outputs', 'cam_results'), exist_ok=True)

    dataset_test = get_datasets(IMAGE_RESIZE)
    test_loader = get_data_loader(dataset_test)

    checkpoint = torch.load(os.path.join('..', 'outputs', 'model.pth'))
    # Load the model.
    model = build_model(
        pretrained=False,
        fine_tune=False, 
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    predictions_list, ground_truth_list, acc = test(
        model, 
        test_loader,
        DEVICE
    )
    print(f"Test accuracy: {acc:.3f}%")