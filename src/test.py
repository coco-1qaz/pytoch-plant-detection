import torch
import numpy as np
import cv2
import os
import torch.nn.functional as F
import torchvision.transforms as transforms

from model import build_model

# Constants and other configurations.
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

def denormalize(
    x, 
    mean=[0.485, 0.456, 0.406], 
    std=[0.229, 0.224, 0.225]
):
    for t, m, s in zip(x, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(x, 0, 1)

def test(model, device, test_result_save_dir=None):
    """
    Function to test the trained model on live video stream.

    :param model: The trained model.
    :param device: The computation device.
    :param test_result_save_dir: Directory to save test results (optional).

    Returns:
        None
    """
    model.eval()
    print('Starting live video stream...')
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (webcam)
    
    if test_result_save_dir:
        os.makedirs(test_result_save_dir, exist_ok=True)

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame.
            frame = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(frame, (IMAGE_RESIZE, IMAGE_RESIZE))
            image = transforms.ToTensor()(image)
            image = image.unsqueeze(0).to(device)

            # Forward pass.
            outputs = model(image)
            # Softmax probabilities.
            predictions = torch.softmax(outputs, dim=1).cpu().numpy()
            # Predicted class number.
            output_class = np.argmax(predictions)

            # Format the probability to display with two decimal places.
            prob = '{:.2f}%'.format(predictions[0, output_class] * 100)

            # Display the class name and probability on the frame.
            class_name = CLASS_NAMES[output_class]
            if class_name in ['Powdery', 'Rust']:
                class_name = 'Unhealthy'
            text = f'{class_name}: {prob}'
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Show the frame with predictions.
            cv2.imshow('Live Video Stream', frame)

            if test_result_save_dir:
                save_path = os.path.join(test_result_save_dir, f'frame_{counter}.jpg')
                cv2.imwrite(save_path, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    checkpoint = torch.load(os.path.join('..', 'outputs', 'model.pth'))
    # Load the model.
    model = build_model(
        pretrained=False,
        fine_tune=False, 
        num_classes=len(CLASS_NAMES)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    test(model, DEVICE)

