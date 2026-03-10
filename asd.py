import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from torch import nn


df=pd.read_csv("03-\\08-email_classification_svm.csv")

# EKLENDİ: Veri seti görselleştirme
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=df["subject_formality_score"],
    y=df["sender_relationship_score"],
    hue=df["email_type"],
    palette="coolwarm"
)
plt.title("Email Classification Dataset")
plt.xlabel("Subject Formality Score")
plt.ylabel("Sender Relationship Score")
plt.show()

#sns.scatterplot(x=df["subject_formality_score"],y=df["sender_relationship_score"],hue=df["email_type"])
#plt.show()

X=df[['subject_formality_score','sender_relationship_score']].values
y=df["email_type"].values

X_train,X_test,y_train,y_test= train_test_split(X,y, test_size=0.2, random_state=42)

X_train=torch.tensor(X_train, dtype=torch.float32)
X_test=torch.tensor(X_test, dtype=torch.float32)

y_train=torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test=torch.tensor(y_test,dtype=torch.float32).unsqueeze(1)

class ClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1=nn.Linear(in_features=2,out_features=5)
        self.layer2=nn.Linear(in_features=5,out_features=1)

    def forward(self,x):
        return self.layer2(self.layer1(x))
    
model0=ClassificationModel()
loss_fn=nn.BCEWithLogitsLoss()
optimizer=torch.optim.SGD(params=model0.parameters(),lr=0.01)

def calculate_acc(y_predic,y_test):
    correct=torch.eq(y_test,y_predic).sum().item()
    acc=(correct/len(y_predic))*100
    return acc 

y_logits=model0(X_test)[:5]

y_pred_probs=torch.sigmoid(y_logits)

y_preds= torch.round(y_pred_probs)


calculate_acc(y_test[:5],y_preds[:5])


torch.manual_seed(42)
epochs=100

# EKLENDİ: loss ve accuracy değerlerini saklamak için listeler
train_losses = []
test_losses = []
train_accs = []
test_accs = []

for epoch in range(epochs):

    model0.train()
    y_logits=model0(X_train)

    loss=loss_fn(y_logits,y_train)  
    y_pred = torch.round(torch.sigmoid(y_logits))
    acc = calculate_acc(y_predic=y_pred, y_test=y_train)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    model0.eval()
    with torch.inference_mode():
        test_logits=model0(X_test)
        test_pred=torch.round(torch.sigmoid(test_logits))

        test_loss=loss_fn(test_logits,y_test)
        test_acc=calculate_acc(y_test=y_test,y_predic=test_pred)

        # EKLENDİ: her epoch'taki skorları kaydet
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        train_accs.append(acc)
        test_accs.append(test_acc)

        if epoch %5==0:
            print(f"epoch {epoch} loss {loss.item():.4f} acc {acc:.2f} test loss {test_loss.item():.4f} test acc {test_acc:.2f}")

# EKLENDİ: Loss grafiği
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")
plt.title("Training vs Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

# EKLENDİ: Accuracy grafiği
plt.figure(figsize=(8, 5))
plt.plot(train_accs, label="Train Accuracy")
plt.plot(test_accs, label="Test Accuracy")
plt.title("Training vs Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid(True)
plt.show()

# EKLENDİ: Final skorları yazdır
print("\nFinal Scores:")
print(f"Final Train Loss: {train_losses[-1]:.4f}")
print(f"Final Test Loss: {test_losses[-1]:.4f}")
print(f"Final Train Accuracy: {train_accs[-1]:.2f}%")
print(f"Final Test Accuracy: {test_accs[-1]:.2f}%")

# EKLENDİ: Decision Boundary görselleştirmesi
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = torch.meshgrid(
    torch.linspace(x_min, x_max, 200),
    torch.linspace(y_min, y_max, 200),
    indexing="ij"
)

grid = torch.stack((xx.flatten(), yy.flatten()), dim=1)

model0.eval()
with torch.inference_mode():
    grid_logits = model0(grid)
    grid_preds = torch.round(torch.sigmoid(grid_logits))

Z = grid_preds.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx.numpy(), yy.numpy(), Z.numpy(), alpha=0.3, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolors="k")
plt.title("Decision Boundary")
plt.xlabel("Subject Formality Score")
plt.ylabel("Sender Relationship Score")
plt.show()