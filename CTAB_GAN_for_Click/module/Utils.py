import numpy as np
import torch
import torch.utils.data
from torch.nn import functional as F
from torch.nn import (LeakyReLU, ReLU, Conv2d, ConvTranspose2d, BatchNorm2d, Sigmoid, init)


def random_choice_prob_index_sampling(probs, col_idx):
    """
    Used to sample a specific category within a chosen one-hot-encoding representation

    Inputs:
    1) probs -> probability mass distribution of categories
    2) col_idx -> index used to identify any given one-hot-encoding

    Outputs:
    1) option_list -> list of chosen categories

    """

    option_list = []
    for i in col_idx:
        # for improved stability
        pp = probs[i] + 1e-6
        pp = pp / sum(pp)
        # sampled based on given probability mass distribution of categories within the given one-hot-encoding
        option_list.append(np.random.choice(np.arange(len(probs[i])), p=pp))

    return np.array(option_list).reshape(col_idx.shape)


def cond_loss(data, output_info, c, m):
    """
    Used to compute the conditional loss for ensuring the generator produces the desired category as specified
    by the conditional vector

    Inputs:
    1) data -> raw data synthesized by the generator
    2) output_info -> column information corresponding to the data transformer
    3) c -> conditional vectors used to synthesize a batch of data
    4) m -> a matrix to identify chosen one-hot-encodings across the batch

    Outputs:
    1) loss -> conditional loss corresponding to the generated batch

    """

    # used to store cross entropy loss between conditional vector and all generated one-hot-encodings
    tmp_loss = []
    # counter to iterate generated data columns
    st = 0
    # counter to iterate conditional vector
    st_c = 0
    # iterating through column information
    for item in output_info:
        # ignoring numeric columns
        if item[1] == 'tanh':
            st += item[0]
            continue
        # computing cross entropy loss between generated one-hot-encoding and corresponding
        # encoding of conditional vector
        elif item[1] == 'softmax':
            ed = st + item[0]
            ed_c = st_c + item[0]
            tmp = F.cross_entropy(
                data[:, st:ed],
                torch.argmax(c[:, st_c:ed_c], dim=1),
                reduction='none')
            tmp_loss.append(tmp)
            st = ed
            st_c = ed_c

    # computing the loss across the batch only and only for the relevant one-hot-encodings by applying the mask
    tmp_loss = torch.stack(tmp_loss, dim=1)
    loss = (tmp_loss * m).sum() / data.size()[0]

    return loss


def get_st_ed(target_col_index, output_info):
    """
    Used to obtain the start and ending positions of the target column as per the transformed data to be used by the
    classifier

    Inputs:
    1) target_col_index -> column index of the target column used for machine learning tasks
    (binary/multi-classification) in the raw data
    2) output_info -> column information corresponding to the data after applying the data transformer

    Outputs:
    1) starting (st) and ending (ed) positions of the target column as per the transformed data

    """
    # counter to iterate through columns
    st = 0
    # counter to check if the target column index has been reached
    c = 0
    # counter to iterate through column information
    tc = 0
    # iterating until target index has reached to obtain starting position of the one-hot-encoding used to represent
    # target column in transformed data
    for item in output_info:
        # exiting loop if target index has reached
        if c == target_col_index:
            break
        if item[1] == 'tanh':
            st += item[0]
        elif item[1] == 'softmax':
            st += item[0]
            c += 1
        tc += 1

        # obtaining the ending position by using the dimension size of the one-hot-encoding
        # used to represent the target column
    ed = st + output_info[tc][0]

    return st, ed


def determine_layers_disc(side, num_channels):
    """
    This function describes the layers of the discriminator network as per
    DCGAN (https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

    Inputs:
    1) side -> height/width of the input fed to the discriminator
    2) num_channels -> no. of channels used to decide the size of respective hidden layers

    Outputs:
    1) layers_D -> layers of the discriminator network

    """

    # computing the dimensionality of hidden layers
    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        # the number of channels increases by a factor of 2 whereas the height/width decreases
        # by the same factor with each layer
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    # constructing the layers of the discriminator network based on the recommendations mentioned
    # in https://arxiv.org/abs/1511.06434
    layers_D = []
    for prev, curr in zip(layer_dims, layer_dims[1:]):
        layers_D += [
            Conv2d(prev[0], curr[0], 4, 2, 1, bias=False),
            BatchNorm2d(curr[0]),
            LeakyReLU(0.2, inplace=True)
        ]
    # last layer reduces the output to a single numeric value which is squashed to a probability
    # using sigmoid function
    layers_D += [
        Conv2d(layer_dims[-1][0], 1, layer_dims[-1][1], 1, 0),
        Sigmoid()
    ]

    return layers_D


def determine_layers_gen(side, random_dim, num_channels):
    """
    This function describes the layers of the generator network

    Inputs:
    1) random_dim -> height/width of the noise matrix to be fed for generation
    2) num_channels -> no. of channels used to decide the size of respective hidden layers

    Outputs:
    1) layers_G -> layers of the generator network

    """

    # computing the dimensionality of hidden layers
    layer_dims = [(1, side), (num_channels, side // 2)]

    while layer_dims[-1][1] > 3 and len(layer_dims) < 4:
        layer_dims.append((layer_dims[-1][0] * 2, layer_dims[-1][1] // 2))

    # similarly constructing the layers of the generator network based on the recommendations mentioned
    # in https://arxiv.org/abs/1511.06434
    # first layer of the generator takes the channel dimension of the noise matrix to the desired maximum
    # channel size of the generator's layers
    layers_G = [
        ConvTranspose2d(
            random_dim, layer_dims[-1][0], layer_dims[-1][1], 1, 0, output_padding=0, bias=False)
    ]

    # the following layers are then reversed with respect to the discriminator
    # such as the no. of channels reduce by a factor of 2 and height/width of generated image
    # increases by the same factor with each layer
    for prev, curr in zip(reversed(layer_dims), reversed(layer_dims[:-1])):
        layers_G += [
            BatchNorm2d(prev[0]),
            ReLU(True),
            ConvTranspose2d(prev[0], curr[0], 4, 2, 1, output_padding=0, bias=True)
        ]

    return layers_G


def apply_activate(data, output_info):
    """
    This function applies the final activation corresponding to the column information associated with transformer

    Inputs:
    1) data -> input data generated by the model in the same format as the transformed input data
    2) output_info -> column information associated with the transformed input data

    Outputs:
    1) act_data -> resulting data after applying the respective activations

    """

    data_t = []
    # used to iterate through columns
    st = 0
    # used to iterate through column information
    for item in output_info:
        # for numeric columns a final tanh activation is applied
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(torch.tanh(data[:, st:ed]))
            st = ed
        # for one-hot-encoded columns, a final gumbel softmax (https://arxiv.org/pdf/1611.01144.pdf) is used
        # to sample discrete categories while still allowing for back propagation
        elif item[1] == 'softmax':
            ed = st + item[0]
            # note that as tau approaches 0, a completely discrete one-hot-vector is obtained
            data_t.append(F.gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed

    act_data = torch.cat(data_t, dim=1)

    return act_data


def weights_init(model):
    """
    This function initializes the learnable parameters of the convolutional and batch norm layers

    Inputs:
    1) model->  network for which the parameters need to be initialized

    Outputs:
    1) network with corresponding weights initialized using the normal distribution

    """

    class_name = model.__class__.__name__

    if class_name.find('Conv') != -1:
        init.normal_(model.weight.data, 0.0, 0.02)

    elif class_name.find('BatchNorm') != -1:
        init.normal_(model.weight.data, 1.0, 0.02)
        init.constant_(model.bias.data, 0)
