import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import r2_score

df = pd.read_csv("06-study_hours_grades.csv")

# Veri ön işleme
X = torch.tensor(df["study_hours"].values, dtype=torch.float32)
y = torch.tensor(df["grade"].values, dtype=torch.float32)

train_split=int(len(X)*0.8)

X_train,y_train=X[:train_split],y[:train_split]

X_test, y_test=X[train_split:],y[train_split:]



class SimpleLinearReg(nn.Module):
    def __init__(self):
        super().__init__()

        self.weights=nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)
        self.bias=nn.Parameter(torch.randn(1,dtype=torch.float),requires_grad=True)

    def forward(self,x: torch.Tensor):
        return self.weights*x +self.bias
    

torch.manual_seed(42)
model0=SimpleLinearReg()


with torch.inference_mode():
    y_pred=model0(X_test)
    print(y_pred)

loss_fn=nn.MSELoss() #mse
optimizer=torch.optim.SGD(params=model0.parameters(), lr=0.01) #0.01
torch.manual_seed(42)
epochs=120
train_loss_val=[]
test_loss_val=[]
epochs_count=[]

for epoch in range(epochs):
    model0.train()
    y_pred=model0(X_train)
    loss=loss_fn(y_pred,y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model0.eval()
    
    with torch.inference_mode():
        test_pred=model0(X_test)

        test_loss=loss_fn(test_pred,y_test.type(torch.float))

        if epoch %5==0:
            epochs_count.append(epoch)
            train_loss_val.append(loss.detach().numpy())
            test_loss_val.append(test_loss.detach().numpy())
            print(f"epochs {epoch}, train loss {loss}, test loss {test_loss}")

model0.eval()
with torch.inference_mode():
    y_pred=model0(X_test)

r2 = r2_score(y_test.numpy(), y_pred.numpy())
mse = loss_fn(y_pred, y_test).item()

print("\nModel Performans Skorları:")
print(f"R² Score: {r2:.4f}")
print(f"Mean Squared Error: {mse:.4f}")
plt.scatter(X_train,y_train, c="b",s=6,label="training data")
plt.scatter(X_test,y_test,c="r",s=6,label="test data")
plt.scatter(X_test,y_pred.detach().numpy(),c="g",s=6,label="predictions")
plt.legend()


plt.text(0.05, 0.95, f'R² Score: {r2:.4f}\nMSE: {mse:.4f}', 
         transform=plt.gca().transAxes, fontsize=12, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.title("Sınav Linear Reg")
plt.xlabel("Sınava çalışma süresi")
plt.ylabel("Sınavdan alınan not")
plt.get_current_fig_manager().set_window_title("Sınav Linear Reg")
plt.show()
