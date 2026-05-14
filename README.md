# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
The aim of this experiment is to develop a Convolutional Neural Network (CNN) model to classify images into different categories. The model takes image data as input, learns important features using convolutional layers, and predicts the correct class label. A dataset of labeled images (such as handwritten digits from 0 to 9) is used to train and test the model. Finally, the performance of the model is evaluated, and it is used to predict the class of new unseen images.

## Neural Network Model
<img width="835" height="427" alt="image" src="https://github.com/user-attachments/assets/eaed7ea2-019b-40a8-99b5-34f8b79b91a4" />


## DESIGN STEPS
### STEP 1: 
Load the dataset from the tensorflow library.

### STEP 2: 
Preprocess the dataset.



### STEP 3: 
Create and train your model.
### STEP 4: 
Include the training loss, validation loss vs iteration plot.
### STEP 5: 
Test the model for your handwritten scanned images.
### STEP 6: 
Create and train your model.

## PROGRAM

### Name:sharukesh s

### Register Number: 212224220095

```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        # write your code here
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)

        return x

from torchsummary import summary

# Initialize model
model = CNNClassifier()

# Move model to GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)

# Print model summary
print('Name: sharukesh s')
print('Register Number: 212224220095')
summary(model, input_size=(1, 28, 28))

# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):
  print('Name: sharukesh s')
  print('Register Number: 212224220095')
  for epoch in range(num_epochs):
      model.train()
      running_loss = 0.0
      for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

      print('Name:sharukesh s ')
      print('Register Number:212224220095 ')
      print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')


# Train the model
train_model(model, train_loader)
```

### OUTPUT

## Training Loss per Epoch
<img width="314" height="196" alt="Screenshot 2026-05-14 080501" src="https://github.com/user-attachments/assets/8a937129-a264-4ccf-9bd4-c96c7e3fdbd8" />


## Confusion Matrix
<img width="826" height="748" alt="Screenshot 2026-05-14 080411" src="https://github.com/user-attachments/assets/a874887e-7d86-4d8c-9a62-e1b0a3c8273b" />


## Classification Report
<img width="514" height="400" alt="Screenshot 2026-05-14 080442" src="https://github.com/user-attachments/assets/eb982a4b-885c-4e83-a735-03929c0ecbe4" />


## New Sample Data Prediction
<img width="482" height="527" alt="Screenshot 2026-05-14 080422" src="https://github.com/user-attachments/assets/1c374947-d3cd-4465-b8e8-11621c0835db" />



## RESULT
The Convolutional Neural Network (CNN) model was successfully trained and achieved good classification performance on the given image dataset.
