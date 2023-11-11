import torch


class ImageTransformer:
    """
    Transformer responsible for translating data rows to images and vice versa

    Variables:
    1) side -> height/width of the image

    Methods:
    1) __init__() -> initializes image transformer object with given input
    2) transform() -> converts tabular data records into square image format
    3) inverse_transform() -> converts square images into tabular format

    """

    def __init__(self, side):
        self.height = side

    def transform(self, data):
        if self.height * self.height > len(data[0]):
            # tabular data records are padded with 0 to conform to square shaped images
            padding = torch.zeros((len(data), self.height * self.height - len(data[0]))).to(data.device)
            data = torch.cat([data, padding], dim=1)

        return data.view(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        data = data.view(-1, self.height * self.height)
        return data
