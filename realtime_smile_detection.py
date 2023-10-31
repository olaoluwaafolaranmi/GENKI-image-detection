import cv2
import torch
import matplotlib.pyplot as plt
from lenet5_3d import LeNet5_3D
import numpy as np

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Get the pretrained model
lenet5_3d = LeNet5_3D()
lenet5_3d.load_state_dict(torch.load(f'model/lenet5.pt'), False)

while True:
    # Read the frame
    ret, frame = cap.read()

    # Perform image processing here
    # Get the square
    height, width, channels = frame.shape
    square_size = min(height, width)
    x = int((width - square_size) / 2)
    y = int((height - square_size) / 2)
    square = frame[y:y+square_size, x:x+square_size]

    # Resize the square frame
    resized_square = cv2.resize(square, (64, 64))
    resized_square = cv2.cvtColor(resized_square, cv2.COLOR_BGR2RGB)
    resized_square = resized_square.transpose([2,0,1])
    resized_square = resized_square[np.newaxis, ...]

    # Put the image through the network
    resized_square = torch.from_numpy(resized_square).float()
    lenet5_3d.eval()
    # output = 0.5
    output = lenet5_3d(resized_square)
    output = torch.sigmoid(output)
    output = output.detach().numpy()[0][0]
   
    # Display the resulting frame
    if output > 0.5:
        cv2.putText(frame, f"Smiling, probability: {output:.3f}", (50, 670), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 255, 0),thickness = 2, lineType=2)
    else:
        cv2.putText(frame, f"Not smiling, probability: {output:.3f}", (50, 670), cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, color = (0, 0, 255),thickness = 2, lineType=2)
    cv2.imshow('Real-time Image Processing', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


# Release the webcam and destroy all windows
cap.release()
cv2.destroyAllWindows()