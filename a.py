import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt 
import torch.nn as nn

df=pd.read_csv("02-\\06-study_hours_grades.csv")


X=torch.tensor(df["study_hours"].values,dtype=torch.float32).unsqueeze(1)
y=torch.tensor(df["grade"].values,dtype=torch.float32).unsqueeze(1)

train_split=int(0.8*len(X))

X_train,y_train=X[:train_split],y[:train_split]
X_test,y_test=X[train_split:],y[train_split:]


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer=nn.Linear(in_features=1,out_features=1,bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)
    
torch.manual_seed(42)
model=LinearRegressionModel()

loss_fn=nn.MSELoss()

optimizer=torch.optim.SGD(params=model.parameters(), lr=0.001)
epochs=120

train_losses=[]
test_losses=[]

model.train()
for epoch in range(epochs):
    y_pred=model(X_train)
    loss=loss_fn(y_pred,y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    train_losses.append(loss.item())
    

    model.eval()
    with torch.inference_mode():
        test_pred_temp=model(X_test)
        test_loss_temp=loss_fn(test_pred_temp,y_test)
        test_losses.append(test_loss_temp.item())
    model.train()

    if epoch %10==0:
        print(f"epoch {epoch} | train loss {loss.item():.4f} | test loss {test_loss_temp.item():.4f}")


model.eval()
with torch.inference_mode():
    train_pred=model(X_train)
    test_pred=model(X_test)
    train_loss=loss_fn(train_pred,y_train).item()
    test_loss=loss_fn(test_pred,y_test).item()
    
    def r2_score(y_true,y_pred):
        ss_res=torch.sum((y_true-y_pred)**2)
        ss_tot=torch.sum((y_true-torch.mean(y_true))**2)
        return (1-(ss_res/ss_tot)).item()
    
    train_r2=r2_score(y_train,train_pred)
    test_r2=r2_score(y_test,test_pred)
    
  
    train_rmse=torch.sqrt(torch.tensor(train_loss)).item()
    test_rmse=torch.sqrt(torch.tensor(test_loss)).item()

print(f"\n=== Final Results ===")
print(f"Train MSE: {train_loss:.4f} | Train RMSE: {train_rmse:.4f} | Train R²: {train_r2:.4f}")
print(f"Test MSE: {test_loss:.4f} | Test RMSE: {test_rmse:.4f} | Test R²: {test_r2:.4f}")

# Plot 1: Loss Curve
fig,axes=plt.subplots(2,2,figsize=(12,10))

axes[0,0].plot(train_losses,label="Train Loss",color="blue",linewidth=2)
axes[0,0].plot(test_losses,label="Test Loss",color="red",linewidth=2)
axes[0,0].set_xlabel("Epoch")
axes[0,0].set_ylabel("Loss (MSE)")
axes[0,0].set_title("Model Loss Curve")
axes[0,0].legend()
axes[0,0].grid(True,alpha=0.3)

# Plot 2: Predictions vs Actual (Train)
axes[0,1].scatter(X_train,y_train,c="blue",label="Actual",s=20,alpha=0.6)
axes[0,1].scatter(X_train,train_pred.detach(),c="red",label="Predicted",s=20,alpha=0.6)
axes[0,1].plot(X_train,train_pred.detach(),c="red",linewidth=2,alpha=0.5)
axes[0,1].set_xlabel("Study Hours")
axes[0,1].set_ylabel("Grade")
axes[0,1].set_title(f"Training Data (R²={train_r2:.4f})")
axes[0,1].legend()
axes[0,1].grid(True,alpha=0.3)

# Plot 3: Predictions vs Actual (Test)
axes[1,0].scatter(X_test,y_test,c="green",label="Actual",s=20,alpha=0.6)
axes[1,0].scatter(X_test,test_pred.detach(),c="orange",label="Predicted",s=20,alpha=0.6)
axes[1,0].plot(X_test,test_pred.detach(),c="orange",linewidth=2,alpha=0.5)
axes[1,0].set_xlabel("Study Hours")
axes[1,0].set_ylabel("Grade")
axes[1,0].set_title(f"Test Data (R²={test_r2:.4f})")
axes[1,0].legend()
axes[1,0].grid(True,alpha=0.3)

# Plot 4: Residuals (Test)
residuals=(y_test-test_pred.detach()).numpy()
axes[1,1].scatter(test_pred.detach(),residuals,c="purple",s=30,alpha=0.6)
axes[1,1].axhline(y=0,color="black",linestyle="--",linewidth=1)
axes[1,1].set_xlabel("Predicted Values")
axes[1,1].set_ylabel("Residuals")
axes[1,1].set_title("Residuals Plot (Test)")
axes[1,1].grid(True,alpha=0.3)

plt.tight_layout()
plt.show()
