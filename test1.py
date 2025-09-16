import os
from skimage import io, transform, img_as_float
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

from dataset import RescaleT, ToTensorLab, SalObjDataset
from model.PAFR import Net


# === Metrics ===
def Fmeasure(pred, gt):
    beta2 = 0.3
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    pred_bin = (pred >= 0.5).astype(np.float32)
    gt = gt.astype(np.float32)

    tp = np.sum(pred_bin * gt)
    precision = tp / (np.sum(pred_bin) + 1e-8)
    recall = tp / (np.sum(gt) + 1e-8)
    fmeasure = (1 + beta2) * precision * recall / (beta2 * precision + recall + 1e-8)
    return fmeasure


def MAE_metric(pred, gt):
    return np.mean(np.abs(pred.astype(np.float32) - gt.astype(np.float32)))


def S_measure(pred, gt):
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = gt.astype(np.float32)
    alpha = 0.5
    y = np.mean(gt)
    if y == 0:
        return 1 - np.mean(pred)
    elif y == 1:
        return np.mean(pred)
    else:
        Q = alpha * np.mean((pred - np.mean(pred)) * (gt - y)) / (np.std(pred) * np.std(gt) + 1e-8) + \
            (1 - alpha) * (1 - np.mean(np.abs(pred - gt)))
        return Q


def E_measure(pred, gt):
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt = gt.astype(np.float32)
    return 1 - np.mean(np.abs(pred - gt))


# === Save output ===
def save_output(image_name, pred, d_dir):
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    pred = (pred * 255).astype(np.uint8)
    im = Image.fromarray(pred).convert('RGB')

    img_name = os.path.basename(image_name)
    save_path = os.path.join(d_dir, img_name.replace('.jpg', '.png'))

    im.save(save_path)


if __name__ == '__main__':
    # === Paths ===
    image_dir = '/kaggle/input/eorssd/test-images/'
    gt_dir = '/kaggle/input/eorssd/test-labels/'
    prediction_dir = '/kaggle/working/results/'
    model_dir = '/kaggle/input/pafr/pytorch/default/1/premodel.pth'

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir)

    img_name_list = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]

    # === Dataset & Dataloader ===
    test_salobj_dataset = SalObjDataset(
        img_name_list=[os.path.join(image_dir, f) for f in img_name_list],
        gt_name_list=[],
        mask_name_list=[],
        transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)])
    )
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # === Load model ===
    print("...load model...")
    model = Net(3)
    model.load_state_dict(torch.load(model_dir))
    model.cuda()
    model.eval()

    # === Metrics storage ===
    F_list, S_list, E_list, MAE_list = [], [], [], []

    # === Inference Loop ===
    for i_test, data_test in enumerate(test_salobj_dataloader):
        img_name = img_name_list[i_test]
        print("inferencing:", img_name)

        inputs_test = data_test['image'].type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, da = model(inputs_test)

        pred = d1[:, 0, :, :].squeeze().cpu().data.numpy()

        # Save prediction
        save_output(os.path.join(image_dir, img_name), pred, prediction_dir)

        # Load GT mask
        gt_path = os.path.join(gt_dir, img_name.replace('.jpg', '.png'))
        gt = io.imread(gt_path)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        gt = img_as_float(gt)
        gt = transform.resize(gt, pred.shape, mode='constant')

        # === Metrics ===
        F_list.append(Fmeasure(pred, gt))
        S_list.append(S_measure(pred, gt))
        E_list.append(E_measure(pred, gt))
        MAE_list.append(MAE_metric(pred, gt))

        del d1, d2, d3, d4, d5, d6, da

    # === Final Results ===
    print("\n=== Evaluation Results on EORSSD Test Set ===")
    print("F-measure: {:.4f}".format(np.mean(F_list)))
    print("S-measure: {:.4f}".format(np.mean(S_list)))
    print("E-measure: {:.4f}".format(np.mean(E_list)))
    print("MAE: {:.4f}".format(np.mean(MAE_list)))
