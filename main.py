from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F  # Import the functional module
import io

app = Flask(__name__)
classes = [
    "Mantled Howler",
    "Patas Monkey",
    "Bald Uakari",
    "Japanese Macaque",
    "Pygmy Marmoset",
    "White headed Capuchi",
    "Silvery Marmoset",
    "Common Squirrel Monkey",
    "Black Headed Night Monkey",
    "Nilgiri Langur",
]
# Load your trained model
model = models.resnet18()  # Initialize the architecture
num_ftrs = model.fc.in_features
number_of_classes = 10
model.fc = torch.nn.Linear(num_ftrs, number_of_classes)
model.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
model.eval()

# Image preprocessing
def transform_image(image_bytes):
    my_transforms = transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4363, 0.4328, 0.3291], std=[0.2135, 0.2081, 0.2044])
    ])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        tensor = transform_image(img_bytes)
        outputs = model(tensor)
        _, predicted = torch.max(outputs.data, 1)
        # Apply softmax to get probabilities
        probabilities = F.softmax(outputs.data, dim=1)
        # Get the probability of the predicted class
        confidence_score = probabilities[0][predicted[0]].item()
        print("Sending Response to Client")
        return jsonify({'prediction': classes[int(predicted[0])], 'confidence_score': confidence_score})

if __name__ == '__main__':
    app.run()