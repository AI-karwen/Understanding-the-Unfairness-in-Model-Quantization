import torch
from torch.utils.data import DataLoader, TensorDataset

# 创建一个虚拟数据集
X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))

# 创建数据集和数据加载器
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10)

# 定义模型和损失函数
model = torch.nn.Linear(10, 2)
criterion = torch.nn.CrossEntropyLoss()

# 计算数据集的黑塞矩阵特征值
total_eigenvalues = 0

for batch_x, batch_y in dataloader:
    # 将梯度归零
    model.zero_grad()
    
    # 计算预测值和损失
    y_pred = model(batch_x)
    loss = criterion(y_pred, batch_y)
    
    # 计算二阶导数
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    hessian = []
    for j, (grad, param) in enumerate(zip(grads, model.parameters())):
        row = []
        for k in range(grad.numel()):
            grad2 = torch.autograd.grad(grad.flatten()[k], param, retain_graph=True)[0]
            row.append(grad2.flatten())
        hessian.append(torch.cat(row))
    hessian = torch.cat(hessian)
    
    # 计算特征值
    eigenvalues, _ = torch.eig(hessian, eigenvectors=False)
    eigenvalues = eigenvalues[:, 0]
    total_eigenvalues += eigenvalues.sum().item()

# 打印数据集的平均黑塞矩阵特征值
print("平均黑塞矩阵特征值：", total_eigenvalues / len(dataloader))
