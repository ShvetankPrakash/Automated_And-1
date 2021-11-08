import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from torchvision import transforms, datasets

# Used for debugging
#cap = cv2.VideoCapture("test_data_clips/train_clip_1.mp4")
cap = cv2.VideoCapture("test_data_clips/clip_8.mp4")

if (cap.isOpened() == False):
    print('Error while trying to read video. Plese check again...')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
 
# Could not use this for my implementation as it did not work
#fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
# define codec and create VideoWriter object
#out = cv2.VideoWriter("fouls.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))

# Had to look into OpenCV Docs and replace with this to do it this way instead using example
frames_per_second = 30
video_size = (frame_width, frame_height) # this is the size of my source video
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
out = cv2.VideoWriter()
out.open('output.mov', fourcc, frames_per_second, video_size, True) 

# Used for debugging
#print("BEFORE LOOP")
# I added this to modify the code and make it work for my implementation
data_transforms = transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])
# I added this to modify the code and make it work for my implementation
test_dataset = datasets.ImageFolder(root="./initial_test_dataset", transform=data_transforms)

# Used originally but failed attempt
#test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=1)

# I added this to modify the code and make it work for my implementation
best_model = torch.load("best_model_augmented_dataset_resnet152.pth")

# read until end of video

# Used for debugging
#print(cap.isOpened())

while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        best_model.eval()
        with torch.no_grad():
            # conver to PIL RGB format before predictions
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Could not use this for my implementation as it did not work
            #pil_image = pil_image.permute(0,1)
            #pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
            #pil_image = torch.tensor(pil_image, dtype=torch.float).cuda()
            
            # Had to do it this way instead
            pil_image = data_transforms(pil_image)#aug(image=np.array(pil_image))['image']
            pil_image = pil_image.unsqueeze(0)
            
            outputs = best_model(pil_image)
            _, preds = torch.max(outputs.data, 1)

            # Used for Debugging
            #display_img = torch.squeeze(pil_image, 0)
            #plt.imshow(display_img.permute(1,2,0))
            #plt.show()

# Failed earlier attempt to get this program working for me
############################################################
#             # convert to PIL RGB format before predictions
#             image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             #image = data_transforms_2(image)
#             image = torch.from_numpy(image)
#             #image = data_transforms(image)
#             #print(image.shape)
#             #pil_image = data_transforms(np.array(pil_image))
#             #pil_image = np.transpose(pil_image, (2, 0, 1)).astype(np.float32)
#             #pil_image = torch.tensor(pil_image, dtype=torch.float).cuda()
#             image = image.permute(2,0,1)
#             image = image.unsqueeze(0)            
#             outputs = best_model(image)
#             _, preds = torch.max(outputs.data, 1)
############################################################

            preds = preds.numpy()[0]

        # Used for debugging
        #print(preds)
        #print(outputs)

        # Apply sigmoid to prediction to map to probability / confidence level
        sigmoid = nn.Sigmoid()
        prob = sigmoid(outputs[0][preds]).numpy()
        
        # Used for debugging
        #print(prob)
        #print(prob.numpy())
        
        # Print confidence level always 
        cv2.putText(frame, (test_dataset.classes[preds] + " Confidence: " + str(prob)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        # Modified this to work the way I wanted to for my implementation
        #if preds == 1: 
        #    cv2.putText(frame, test_dataset.classes[preds], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 0), 2)
        cv2.imshow('image', frame)
        out.write(frame)

        # press `q` to exit
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    else: 
        break

# Used for debugging  
#print("AFTER LOOP")

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
