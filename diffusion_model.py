import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import math
import os
from PIL import Image

# mps 설정 확인
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 간단한 UNet 모델
class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64):
        super().__init__()
        
        # 시간 임베딩
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 다운샘플링
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.AvgPool2d(2)
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU(),
            nn.AvgPool2d(2)
        )
        
        # 병목 레이어
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 4),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 4, hidden_channels * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU()
        )
        
        # 업샘플링
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels * 2 + hidden_channels * 2, hidden_channels * 2, 3, padding=1),
            nn.GroupNorm(8, hidden_channels * 2),
            nn.SiLU(),
            nn.Conv2d(hidden_channels * 2, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU()
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(hidden_channels + hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU()
        )
        
        # 최종 출력 레이어
        self.out = nn.Conv2d(hidden_channels, in_channels, 1)
        
        # 시간 임베딩 투영 레이어
        self.time_proj1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.SiLU()
        )
        
        self.time_proj2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 4),
            nn.SiLU()
        )
        
        self.time_proj3 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * 2),
            nn.SiLU()
        )
        
    def forward(self, x, t):
        # 시간 임베딩
        t = t.float().unsqueeze(-1) / 500  # 시간 정규화
        t = self.time_mlp(t)
        
        # 다운샘플링 경로
        x1 = self.down1(x)
        x2 = self.down2(x1)
        
        # 병목 부분
        x_mid = self.bottleneck(x2)
        
        # 업샘플링 경로
        x_up1 = self.up1(torch.cat([x_mid, x2], dim=1))
        x_up2 = self.up2(torch.cat([x_up1, x1], dim=1))
        
        # 최종 출력
        return self.out(x_up2)

# 확산 모델
class DiffusionModel(nn.Module):
    def __init__(self, model, n_steps=500, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        
        # 베타 스케줄
        self.betas = torch.linspace(beta_start, beta_end, n_steps, device=device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        # 샘플링을 위한 계수
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        
    def q_sample(self, x_start, t, noise=None):
        """주어진 시작점 x_start와 타임스텝 t에서 노이즈 추가"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # 인덱싱을 위한 조정
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def forward(self, x_start):
        """학습을 위한 forward pass"""
        batch_size = x_start.shape[0]
        t = torch.randint(0, self.n_steps, (batch_size,), device=device).long()
        noise = torch.randn_like(x_start)
        x_noisy, _ = self.q_sample(x_start, t, noise)
        noise_pred = self.model(x_noisy, t)
        
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def p_sample(self, x, t):
        """샘플링 과정의 단일 스텝"""
        betas_t = self.betas[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1, 1)
        
        # 모델 예측
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )
        
        # t > 0일 때만 노이즈 추가
        if t[0] > 0:
            noise = torch.randn_like(x)
            posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1, 1)
            model_mean = model_mean + torch.sqrt(posterior_variance_t) * noise
        
        return model_mean
    
    @torch.no_grad()
    def sample(self, batch_size=16, img_size=32, channels=3):
        """완전한 역확산 과정을 통한 샘플 생성"""
        shape = (batch_size, channels, img_size, img_size)
        x = torch.randn(shape, device=device)
        
        # 역확산 과정
        for t in tqdm(reversed(range(0, self.n_steps)), total=self.n_steps):
            x = self.p_sample(x, torch.full((batch_size,), t, device=device, dtype=torch.long))
            
        # [-1, 1] 범위로 맞추기
        x = torch.clamp(x, -1, 1)
        # [0, 1] 범위로 변환
        x = (x + 1) / 2
        
        return x

# 통합 모델
class SimpleDiffusion(nn.Module):
    def __init__(self, image_size=32, hidden_channels=64, n_steps=500):
        super().__init__()
        self.image_size = image_size
        
        # UNet 모델
        self.model = SimpleUNet(in_channels=3, hidden_channels=hidden_channels)
        
        # 확산 모델
        self.diffusion = DiffusionModel(
            model=self.model,
            n_steps=n_steps
        )
    
    def forward(self, x):
        """학습 손실 계산"""
        return self.diffusion(x)
    
    def sample(self, batch_size=4):
        """샘플 생성"""
        return self.diffusion.sample(batch_size, self.image_size, 3)

# 학습 함수
def train(model, train_loader, optimizer, epochs, device):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = batch[0].to(device)
            
            optimizer.zero_grad()
            
            # 학습
            loss = model(images)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # 샘플 생성
        if (epoch + 1) % 5 == 0:
            generate_samples(model, epoch)

# 샘플 생성 및 시각화
def generate_samples(model, epoch, num_samples=4):
    model.eval()
    with torch.no_grad():
        # 샘플 생성
        samples = model.sample(batch_size=num_samples)
        
        # 시각화
        samples_cpu = samples.cpu()
        
        plt.figure(figsize=(10, 10))
        for i in range(num_samples):
            plt.subplot(2, 2, i+1)
            plt.imshow(samples_cpu[i].permute(1, 2, 0))
            plt.axis('off')
        
        os.makedirs('samples', exist_ok=True)
        plt.savefig(f"samples/sample_epoch_{epoch+1}.png")
        plt.close()

# 메인 함수
def main():
    # 하이퍼파라미터
    image_size = 32
    batch_size = 16
    epochs = 100
    hidden_channels = 64
    lr = 1e-4
    
    # 데이터셋 준비
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True,
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # 모델 초기화
    model = SimpleDiffusion(
        image_size=image_size,
        hidden_channels=hidden_channels,
        n_steps=500
    ).to(device)
    
    # 옵티마이저
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # 학습 시작
    train(model, train_loader, optimizer, epochs, device)
    
    # 모델 저장
    torch.save(model.state_dict(), "diffusion_model.pth")
    
if __name__ == "__main__":
    main()