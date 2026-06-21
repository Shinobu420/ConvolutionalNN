import torch
from PIL import Image
from convolutional_nn import NeuralNetwork, transform_data

# AMD specific
torch.set_float32_matmul_precision('high')

device = "cuda" if torch.cuda.is_available() else "cpu"


def predict_image(image_path, model):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform_data(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
    prediction_index = torch.argmax(output, dim=1).item()
    classes = ["Cat", "Dog"]
    print(f"Prediction for {image_path}: {classes[prediction_index]}")


if __name__ == "__main__":
    print(f"Using {device} device")

    # 1. Load model
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load("cat_dog_model.pth", map_location=device))
    model.eval()

    # 2. Define list of test images
    test_images = ["./image0.png", "./image1.jpg", "./image2.jpg", "./image3.png", "./image4.png"]

    # 3. Loop through all images
    print("\n--- Starting predictions ---")
    for img_path in test_images:
        predict_image(img_path, model)