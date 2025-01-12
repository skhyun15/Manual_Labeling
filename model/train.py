import os.path
import argparse
import numpy as np
import matplotlib.pyplot as plt
from model_unet import *
from datasets import BoneSegmentDataset
from tqdm import tqdm
from PIL import Image

# our CLI parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--datadir",
    type=str,
    default="/projects3/pi/nhcho/Sev_WBBS/skhyun/segmentation/data/ANT",
    help="directory the BoneScan dataset is in",
)
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument("--num_gpu", type=int, default=1, help="number of gpus")
args = parser.parse_args()


# hyper-parameters (learning rate and how many epochs we will train for)
lr = 0.001
epochs = 500


# cityscapes dataset loading
img_data = BoneSegmentDataset(args.datadir, augment=True)
img_batch = torch.utils.data.DataLoader(
    img_data, batch_size=args.batch_size, shuffle=True, num_workers=4
)


# loss function
# use reconstruction of image if looking to match image output to another image (RGB)
# else if you have a set of classes, we want to do some binary classification on it (cityscapes classes)

recon_loss_func = nn.CrossEntropyLoss()
num_classes = img_data.num_classes


# initiate generator and optimizer
print("creating unet model...")
generator = UnetGenerator(1, img_data.num_classes, 64).cuda()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, weight_decay=1e-5)


# load pretrained model if it is there
file_model = "./unet.pkl"
if os.path.isfile(file_model):
    generator = torch.load(file_model)
    print("    - model restored from file....")
    print("    - filename = %s" % file_model)


# or log file that has the output of our loss
file_loss = open("./unet_loss", "w")


# make the result directory
if not os.path.exists("./result/"):
    os.makedirs("./result/")


def save_in_grid(image, path, num_row=2):
    grid = v_utils.make_grid(
        image,
        nrow=num_row,
    )
    Image.fromarray(grid.permute(1, 2, 0).numpy().astype(np.uint8)).save(path)


# finally!!! the training loop!!!
for epoch in tqdm(range(epochs), desc="epochs"):
    total_loss = 0.0
    for idx_batch, (image, label) in tqdm(
        enumerate(img_batch), desc="batches", total=len(img_batch)
    ):

        # zero the grad of the network before feed-forward
        gen_optimizer.zero_grad()

        # send to the GPU and do a forward pass
        x = image.cuda(0)
        y_label = label.cuda(0)
        y = generator.forward(x)

        # we "squeeze" the groundtruth if we are using cross-entropy loss
        # this is because it expects to have a [N, W, H] image where the values
        # in the 2D image correspond to the class that that pixel should be 0 < pix[u,v] < classes

        y_label = y_label.squeeze(1)
        # assert y_label.size() == (
        #     args.batch_size,
        #     512,
        #     512,
        # ), f"got size of {y_label.size()}"

        # finally calculate the loss and back propagate
        loss = recon_loss_func(y, y_label.long())
        # file에 추가로 작성
        total_loss += loss.item()

        loss.backward()
        gen_optimizer.step()

        # every 400 images, save the current images
        # also checkpoint the model to disk

        if epoch % 10 == 0 and idx_batch == len(img_batch) - 2:
            # nice debug print of this epoch and its loss
            # print("epoch = " + str(epoch) + " | loss = " + str(loss.item()))

            # save the original image and label batches to file
            # 이미지로 변환 (단일 채널 grayscale로)
            normalized_image = torch.zeros(
                (image.size()[0], 3, image.size()[2], image.size()[3])
            )
            for idx in range(0, image.size()[0]):
                each_tensor = image[idx]
                each_normalized = (each_tensor - each_tensor.min()) / (
                    each_tensor.max() - each_tensor.min()
                )
                normalized_image[idx] = each_normalized.repeat(3, 1, 1) * 255

            save_in_grid(normalized_image, f"./result/original_image_{epoch}.png")

            labelrgb = torch.zeros(
                (label.size()[0], 3, label.size()[1], label.size()[2])
            )
            for idx in range(0, label.size()[0]):
                labelrgb[idx] = img_data.class_to_rgb(label[idx])

            save_in_grid(labelrgb, f"./result/label_image_{epoch}.png")

            # max over the classes should be the prediction
            # our prediction is [N, classes, W, H]
            # so we max over the second dimension and take the max response
            # if we are doing rgb reconstruction, then just directly save it to file
            y_threshed = torch.zeros((y.size()[0], 3, y.size()[2], y.size()[3]))
            for idx in range(0, y.size()[0]):
                maxindex = torch.argmax(y[idx], dim=0).cpu().int()
                y_threshed[idx] = img_data.class_to_rgb(maxindex)

            save_in_grid(y_threshed, f"./result/gen_image_{epoch}.png")
    avg_loss = total_loss / len(img_batch)
    file_loss.write(f"{epoch},{avg_loss}\n")
    print(f"epoch = {epoch} | loss = {avg_loss}")
    # finally checkpoint this file to disk
    # torch.save(generator, file_model)

file_loss.close()
