import torch
from torch.nn import Module, Conv2d, Linear, MaxPool2d, ReLU, LogSoftmax, Sigmoid
from torch import flatten, nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
import torch.nn.functional as F

class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3)):
        super().__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.relu = ReLU()
        self.conv2 = Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        output = self.relu(x)
        return output

class Down(torch.nn.Module):
    def __init__(self, in_channels, out_channels, double_conv_kernel_size=(3, 3), max_pool_kernel_size=(2, 2), max_pool_stride=(2,2), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.maxpool = MaxPool2d(kernel_size=max_pool_kernel_size, stride=max_pool_stride)
        self.doubleConv = DoubleConv(in_channels=in_channels, out_channels=out_channels, kernel_size=double_conv_kernel_size)

    def forward(self, x, verbose=False):
        if verbose: print(f"maxpool before: {x.shape}")
        x = self.maxpool(x)
        if verbose: print(f"maxpool after: {x.shape}")
        x = self.doubleConv(x)
        if verbose: print(f"doubleconv after: {x.shape}")
        return x
'''
def crop_to_match(x, target):
    _, _, H, W = target.shape  # Extract target height and width
    _, _, H_x, W_x = x.shape  # Extract input tensor's height and width
    # Compute cropping sizes (symmetric)
    crop_h = (H_x - H) // 2
    crop_w = (W_x - W) // 2
    # Crop symmetrically
    x_cropped = x[:, :, crop_h:crop_h + H, crop_w:crop_w + W]
    return x_cropped
'''
def crop_to_match(x_skip_con, x_expansion_path):
    _, _, H, W = x_expansion_path.shape  # Extract height and width from expansion path layer
    # target is double of the expansion paths deeper layer
    H_target, W_target = 2 * H, 2 * W
    _, _, H_x_skip_con, W_x_skip_con = x_skip_con.shape  # Extract input tensor's height and width
    # Compute cropping sizes (symmetric)
    crop_h = (H_x_skip_con - H_target) // 2
    crop_w = (W_x_skip_con - W_target) // 2
    # Crop symmetrically
    x_cropped = x_skip_con[:, :, crop_h:crop_h + H_target, crop_w:crop_w + W_target]
    return x_cropped

class AttentionGate(torch.nn.Module):
    '''
    g: gating signal from deeper representation
    x: x skip connection
    '''
    def __init__(self, x_channels, g_channels):
        super().__init__()
        self.conv1x1_x = Conv2d(in_channels=x_channels, out_channels=g_channels, kernel_size=(1, 1), stride=(2, 2))
        self.conv1x1_g = Conv2d(in_channels=g_channels, out_channels=g_channels, kernel_size=(1, 1), stride=(1, 1))
        self.relu = ReLU()
        self.conv1x1_psi = Conv2d(in_channels=g_channels, out_channels=1, kernel_size=(1, 1), stride=(1, 1))
        self.sigmoid = Sigmoid()
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


    def forward(self, x, g, verbose=False):
        if verbose: print("attention start")
        if verbose: print(f"x shape {x.shape} - g shape {g.shape}")
        x_conv = self.conv1x1_x(x)
        g_conv = self.conv1x1_g(g)
        if verbose: print(f"after 1x1 conv: x shape {x.shape} - g shape {g.shape}")
        output = self.relu(x_conv + g_conv)
        if verbose: print(f"after relu: {output.shape}")
        output = self.conv1x1_psi(output)
        if verbose: print(f"after psi: {output.shape}")
        output = self.sigmoid(output)
        if verbose: print(f"after sigmoid: {output.shape}")
        output = self.upsample(output)
        if verbose: print(f"after upsample: {output.shape}")
        output = x * output
        if verbose: print(f"attention end: {output.shape}")
        return output


class Up(torch.nn.Module):
    def __init__(self, in_channels, out_channels, double_conv_kernel_size=(3, 3), up_conv_kernel_size=(2, 2), attention_gate=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upConv = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=up_conv_kernel_size, stride=(2,2))
        self.doubleConv = DoubleConv(in_channels=in_channels, out_channels=out_channels, kernel_size=double_conv_kernel_size)
        if attention_gate:
            self.attention_gate = AttentionGate(x_channels=out_channels, g_channels=in_channels)

    def forward(self, x, x_skip_con, verbose=False):
        if verbose: print(f"Up start: x shape {x.shape} - x_skip_con shape {x_skip_con.shape}")
        x_skip_con = crop_to_match(x_skip_con, x)
        if verbose: print(f"Up after crop: x shape {x.shape} - x_skip_con shape {x_skip_con.shape}")
        if self.attention_gate:
            x_skip_con = self.attention_gate(x=x_skip_con, g=x)

        if verbose: print(f"upconv before: {x.shape}")
        x = self.upConv(x)
        if verbose: print(f"upconv after: {x.shape}")
        if verbose: print(f"cat before: {x.shape}")
        x = torch.cat((x, x_skip_con), dim=1)
        if verbose: print(f"cat after: {x.shape}")
        output = self.doubleConv(x)
        if verbose: print(f"after: {output.shape}")
        return output

class UNet(torch.nn.Module):
    def __init__(self, kernel_size=(3, 3), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_ding = DoubleConv(in_channels=1, out_channels=64, kernel_size=kernel_size)

        self.first_deconv = Down(in_channels=64, out_channels=128, double_conv_kernel_size=(3, 3), max_pool_kernel_size=(2, 2), max_pool_stride=(2,2))
        self.second_deconv = Down(in_channels=128, out_channels=256, double_conv_kernel_size=(3, 3), max_pool_kernel_size=(2, 2), max_pool_stride=(2,2))
        self.third_deconv = Down(in_channels=256, out_channels=512, double_conv_kernel_size=(3, 3), max_pool_kernel_size=(2, 2), max_pool_stride=(2,2))
        self.fourth_deconv = Down(in_channels=512, out_channels=1024, double_conv_kernel_size=(3, 3), max_pool_kernel_size=(2, 2), max_pool_stride=(2,2))

        self.first_upconv = Up(in_channels=1024, out_channels=512, double_conv_kernel_size=(3, 3), up_conv_kernel_size=(2,2))
        self.second_upconv = Up(in_channels=512, out_channels=256, double_conv_kernel_size=(3, 3), up_conv_kernel_size=(2,2))
        self.third_upconv = Up(in_channels=256, out_channels=128, double_conv_kernel_size=(3, 3), up_conv_kernel_size=(2,2))
        self.fourth_upconv = Up(in_channels=128, out_channels=64, double_conv_kernel_size=(3, 3), up_conv_kernel_size=(2,2))
        self.self_conv = Conv2d(in_channels=64, out_channels=2, kernel_size=(1,1))
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x, verbose=False):
        # Contracting
        if verbose: print(f"Input: {x.shape}")
        x_down_1 = self.first_ding(x)
        if verbose: print(f"1: {x_down_1.shape}")
        x_down_2 = self.first_deconv(x_down_1)
        if verbose: print(f"2: {x_down_2.shape}")
        x_down_3 = self.second_deconv(x_down_2)
        if verbose: print(f"3: {x_down_3.shape}")
        x_down_4 = self.third_deconv(x_down_3)
        if verbose: print(f"4: {x_down_4.shape}")
        x_down_5 = self.fourth_deconv(x_down_4)
        if verbose: print(f"5: {x_down_5.shape}")

        # Expanding
        x = self.first_upconv(x_down_5, x_down_4)
        if verbose: print(f"4: {x.shape}")
        x = self.second_upconv(x, x_down_3)
        if verbose: print(f"3: {x.shape}")
        x = self.third_upconv(x, x_down_2)
        if verbose: print(f"2: {x.shape}")
        x = self.fourth_upconv(x, x_down_1)
        if verbose: print(f"1: {x.shape}")
        output = self.self_conv(x)
        if verbose: print(f"before interpol: {torch.unique(output)}")
        #output = self.logSoftmax(x)
        # interpolate to actual dimensions
        output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
        if verbose:print(f"after interpol: {torch.unique(output)}")

        if verbose:print(f"output shape {output.shape}")


        # add sth in here?
        return output





class AttentionUNet(torch.nn.Module):
    def __init__(self, kernel_size=(3, 3), *args, **kwargs):
        super().__init__(*args, **kwargs)
        use_attention = True
        self.first_ding = DoubleConv(in_channels=1, out_channels=64, kernel_size=kernel_size)

        self.first_deconv = Down(in_channels=64, out_channels=128, double_conv_kernel_size=(3, 3), max_pool_kernel_size=(2, 2), max_pool_stride=(2,2))
        self.second_deconv = Down(in_channels=128, out_channels=256, double_conv_kernel_size=(3, 3), max_pool_kernel_size=(2, 2), max_pool_stride=(2,2))
        self.third_deconv = Down(in_channels=256, out_channels=512, double_conv_kernel_size=(3, 3), max_pool_kernel_size=(2, 2), max_pool_stride=(2,2))
        self.fourth_deconv = Down(in_channels=512, out_channels=1024, double_conv_kernel_size=(3, 3), max_pool_kernel_size=(2, 2), max_pool_stride=(2,2))

        self.first_upconv = Up(in_channels=1024, out_channels=512, double_conv_kernel_size=(3, 3), up_conv_kernel_size=(2,2), attention_gate=use_attention)
        self.second_upconv = Up(in_channels=512, out_channels=256, double_conv_kernel_size=(3, 3), up_conv_kernel_size=(2,2), attention_gate=use_attention)
        self.third_upconv = Up(in_channels=256, out_channels=128, double_conv_kernel_size=(3, 3), up_conv_kernel_size=(2,2), attention_gate=use_attention)
        self.fourth_upconv = Up(in_channels=128, out_channels=64, double_conv_kernel_size=(3, 3), up_conv_kernel_size=(2,2), attention_gate=use_attention)
        self.self_conv = Conv2d(in_channels=64, out_channels=2, kernel_size=(1,1))
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x, verbose=False):
        # Contracting
        if verbose: print(f"Input: {x.shape}")
        x_down_1 = self.first_ding(x)
        if verbose: print(f"1: {x_down_1.shape}")
        x_down_2 = self.first_deconv(x_down_1)
        if verbose: print(f"2: {x_down_2.shape}")
        x_down_3 = self.second_deconv(x_down_2)
        if verbose: print(f"3: {x_down_3.shape}")
        x_down_4 = self.third_deconv(x_down_3)
        if verbose: print(f"4: {x_down_4.shape}")
        x_down_5 = self.fourth_deconv(x_down_4)
        if verbose: print(f"5: {x_down_5.shape}")

        # Expanding
        x = self.first_upconv(x_down_5, x_down_4)
        if verbose: print(f"4: {x.shape}")
        x = self.second_upconv(x, x_down_3)
        if verbose: print(f"3: {x.shape}")
        x = self.third_upconv(x, x_down_2)
        if verbose: print(f"2: {x.shape}")
        x = self.fourth_upconv(x, x_down_1)
        if verbose: print(f"1: {x.shape}")
        output = self.self_conv(x)
        if verbose: print(f"before interpol: {torch.unique(output)}")
        #output = self.logSoftmax(x)
        # interpolate to actual dimensions
        output = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
        if verbose:print(f"after interpol: {torch.unique(output)}")

        if verbose:print(f"output shape {output.shape}")


        # add sth in here?
        return output




class CoolDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None, target_transform=None, on_gpu=False):
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
        self.mask_names = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])

    def __len__(self):
        count = 0
        for root_dir, cur_dir, files in os.walk(self.img_dir):
            count += len(files)
        return count

    def __getitem__(self, idx):
        image_path = os.path.join(self.img_dir, self.image_names[idx])
        image = read_image(image_path, mode=ImageReadMode.GRAY)

        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        mask = read_image(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        image = image.float()  # Converts the tensor to float
        #mask = torch.tensor(mask, dtype=torch.long).squeeze(0)
        mask = mask.float()  # Convert to float if needed
        mask = mask.squeeze(0)  # Remove the extra channel
        mask = (mask > 0).long()  # Convert to binary mask with values 0 or 1

        return image, mask

learning_rate = 1e-3
batch_size = 16
epochs = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        #print(f"X shape {X.shape}")
        #print(f"y shape {y.shape}")

        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def main():
    train_imgs_path = r"D:\Martin\thesis\data\processed\dataset_0227_final_fun\train\imgs"
    train_masks_path = r"D:\Martin\thesis\data\processed\dataset_0227_final_fun\train\masks"
    val_imgs_path = r"D:\Martin\thesis\data\processed\dataset_0227_final_fun\val\imgs"
    val_masks_path = r"D:\Martin\thesis\data\processed\dataset_0227_final_fun\val\masks"

    train_dataset = CoolDataset(train_imgs_path, train_masks_path)
    val_dataset = CoolDataset(val_imgs_path, val_masks_path)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    u_net_model = AttentionUNet().to(device)
    print(u_net_model)

    X = torch.rand(1, 1, 572, 572, device=device)  # Add batch and channel dimensions
    logits = u_net_model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(u_net_model.parameters(), lr=learning_rate)

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, u_net_model, loss_fn, optimizer)
        test_loop(test_dataloader, u_net_model, loss_fn)
    print("Done!")

if __name__ == "__main__":
    main()
