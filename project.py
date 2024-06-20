import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

# Load a pre-trained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

model.eval()

# Define the image transformation
transform = transforms.Compose([
    transforms.ToTensor()
])

# CATEGORY NAMES from COCO(Common Object in Context) Dataset
class_mapping = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
    'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def main():
    st.title("Image Component Detection")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png","jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width = True)
        
        if st.button('Analyse Image'):
            st.write("Analysing image...")

            image_tensor = transform(image) # Convert the uploaded image to a tensor with a range from [0.0,1.0] and make the channel is the first dimension (c*h*w)
            image_tensor = image_tensor.unsqueeze(0)  # add another dimension

            with torch.no_grad():
                outputs = model(image_tensor) # [{'boxes': tensor([[x1, y1, x2, y2], ...]), 'labels': tensor([1, 2, ...]), 'scores': tensor([0.98, 0.87, ...]) }]
            
            labels = outputs[0]['labels'] # [1,7, ... ]
            scores = outputs[0]['scores'] # [0.98, 0.87, ... ]
            detected_names = set()

            for label, score in zip(labels, scores):
                if score > 0.5:
                    detected_names.add(class_mapping[label.item()])
            
            st.write("Detected components:")
            for name in detected_names:
                st.write(name)

if __name__ == '__main__':
    main()
