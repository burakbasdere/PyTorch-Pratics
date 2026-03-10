import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from torch import nn

# CPU kullanımı
device = torch.device("cpu")

df=pd.read_csv("04-\\08-seismic_activity_svm.csv")

X=df[['underground_wave_energy','vibration_axis_variation']].values
y=df['seismic_event_detected'].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

X_train=torch.tensor(X_train,dtype=torch.float32).to(device)
X_test=torch.tensor(X_test,dtype=torch.float32).to(device)
y_train=torch.tensor(y_train,dtype=torch.float32).unsqueeze(1).to(device)
y_test=torch.tensor(y_test,dtype=torch.float32).unsqueeze(1).to(device)

print("X_train.shape",X_train.shape, 
      "X_test shape", X_test.shape, 
      "y_train shape", y_train.shape, 
      "y_test shape",y_test.shape)

class ClassificanNonLinearModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1=nn.Linear(in_features=2,out_features=10)
        self.layer2=nn.Linear(in_features=10,out_features=10)
        self.layer3=nn.Linear(in_features=10,out_features=1)
        self.relu=nn.ReLU()

    def forward(self,x):
        return self.layer3(self.relu((self.layer2(self.relu(self.layer1(x))))))

model1=ClassificanNonLinearModel().to(device)

loss_fn=nn.BCEWithLogitsLoss()

optimizer=torch.optim.Adam(params=model1.parameters(), lr=0.001)
    
def calculate_acc(y_predic,y_test):
    correct=torch.eq(y_test,y_predic).sum().item()
    acc=(correct/len(y_predic))*100
    return acc 

torch.manual_seed(42)

epochs=400

# Grafik için kayıt listeleri
train_losses=[]
test_losses=[]
train_accs=[]
test_accs=[]

for epoch in range(epochs):

    model1.train()

    y_logits=model1(X_train)
    y_pred=torch.round(torch.sigmoid(y_logits))

    loss=loss_fn(y_logits,y_train)
    acc=calculate_acc(y_test=y_train,y_predic=y_pred)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model1.eval()

    with torch.inference_mode():
        test_logits=model1(X_test)
        test_pred=torch.round(torch.sigmoid(test_logits))
        test_loss=loss_fn(test_logits,y_test)
        test_acc=calculate_acc(y_test=y_test,y_predic=test_pred)

    # değerleri kaydet
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())
    train_accs.append(acc)
    test_accs.append(test_acc)

    if epoch %40==0:
        print(f" epoch:{epoch}, loss:{loss} acc:{acc} test loss {test_loss} test acc {test_acc}")

# -------------------------------
# GRAFİKLER
# -------------------------------

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(train_losses,label="Train Loss")
plt.plot(test_losses,label="Test Loss")
plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs,label="Train Accuracy")
plt.plot(test_accs,label="Test Accuracy")
plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.show()