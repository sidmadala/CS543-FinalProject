import torch
from torchvision.io import read_image
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize, pad
from PIL import Image
import numpy as np

class MIMIC_CXR_Dataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, outcome_list, image_prefix="data/images", scaling_factor=9, tgrt_img_dims=[512,512], transforms=None):
        self.annotation_file = annotation_file
        self.image_prefix = image_prefix
        self.outcome_list = outcome_list
        self.scaling_factor = scaling_factor
        self.tgrt_img_dims = tgrt_img_dims
        self.transforms = transforms

    def __len__(self):
        return len(self.annotation_file)

    def __getitem__(self, image_name):
        ## Get the patient outcomes
        patient_info = self.annotation_file.loc[self.annotation_file['dicom_id'] == image_name]
        label = torch.tensor(patient_info[self.outcome_list].values[:]).squeeze(0)

        ## Resize the input image by a factor of 9 --> want to preserve the image resolution
        im = read_image(f"{self.image_prefix}/{image_name}.jpg")
        if self.transforms is None:
            ## Read the input image
            dims = np.ceil(np.array(im.shape[1:])/self.scaling_factor).astype(int)
            resized_image = resize(im, list(dims))[0]

            ## Ensure that the image is evenly divisible
            resized_image = resized_image[resized_image.shape[0]%2:, resized_image.shape[1]%2:]
            dims = np.array(resized_image.shape)

            ## Pad the image to the required target image size
            transformed_image = pad(resized_image, list(np.flip(np.ceil((np.array(self.tgrt_img_dims) - dims)/2).astype(int)))).unsqueeze(0)
        else:
            im = im[0].numpy()
#            im = np.dstack([im, im, im]).transpose(2,0,1)
            im = np.dstack([im, im, im])
            transformed_image = self.transforms(Image.fromarray(im))

        return transformed_image, label
