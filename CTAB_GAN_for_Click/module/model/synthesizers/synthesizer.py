import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.optim as optim
import json
from torch.optim import Adam

from torch.nn import (BCELoss, CrossEntropyLoss)
from module.model.transformers.transformer import DataTransformer
from module.model.transformers.image_transformer import ImageTransformer
from tqdm import tqdm
from module.Utils import cond_loss, get_st_ed, determine_layers_disc, determine_layers_gen, apply_activate, weights_init
from module.model.synthesizers.sampler import ConditionalVectorSampler, RealDataSampler
from module.model.synthesizers.classifier import Classifier
from module.model.synthesizers.discriminator import Discriminator
from module.model.synthesizers.generator import Generator


class CTABGANSynthesizer:
    """
    This class represents the main model for training and generating synthetic data.
    """

    def __init__(self,
                 class_dim=(256, 256, 256, 256),
                 random_dim=100,
                 num_channels=64,
                 l2scale=1e-5,
                 batch_size=500,
                 epochs=1):
        """
        Initializes the model with user specified parameters.
        """

        self.components = None
        self.output_info = None
        self.model = None
        self.Dtransformer = None
        self.Gtransformer = None
        self.cond_generator = None
        self.transformer = None
        self.random_dim = random_dim
        self.class_dim = class_dim
        self.num_channels = num_channels
        self.dside = None
        self.gside = None
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.generator = None

    def fit(self, train_data=pd.DataFrame, categorical=None, mixed=None, types=None):
        """
        Fits the CTABGANSynthesizer model using the pre-processed training data and associated parameters.
        """

        # Preprocess the training data
        if types is None:
            types = {}
        if mixed is None:
            mixed = {}
        if categorical is None:
            categorical = []
        train_data, target_index, problem_type = self.preprocess_data(train_data, categorical, mixed, types)

        # Initializing the networks and their optimizers
        data_sampler, discriminator, optimizerG, optimizerD, classifier, optimizerC, st_ed = \
            self.initialize_networks(train_data, target_index, problem_type)

        steps_per_epoch = max(1, len(train_data) // self.batch_size)
        # Execute the training
        self.training_loop(train_data, steps_per_epoch, data_sampler, optimizerG, optimizerD, classifier,
                           optimizerC, st_ed, problem_type, discriminator)

    def preprocess_data(self, train_data, categorical, mixed, types):
        problem_type = None
        target_index = None

        if types:
            problem_type = list(types.keys())[0]
            if problem_type:
                target_index = train_data.columns.get_loc(types[problem_type])

        self.transformer = DataTransformer(training_data=train_data, categorical_column=categorical, 
                                           mixed_column=mixed)
        self.transformer.fit()
        del self.transformer

        with open('/content/drive/MyDrive/CTABGANforClickThrough/model.json', 'r') as model_file:
            self.model = json.load(model_file)
        with open('/content/drive/MyDrive/CTABGANforClickThrough/output_info.json', 'r') as output_info_file:
            self.output_info = json.load(output_info_file)
        with open('/content/drive/MyDrive/CTABGANforClickThrough/components.json', 'r') as components_file:
            self.components = json.load(components_file)
        with open('/content/drive/MyDrive/CTABGANforClickThrough/output_dim.json', 'r') as output_dim_file:
            self.output_info = json.load(output_dim_file)

        train_data = self.transformer.transform(train_data.values)

        return train_data, target_index, problem_type

    def initialize_networks(self, train_data, target_index, problem_type):
        data_dim = self.transformer.output_dim
        data_sampler = RealDataSampler(train_data, self.transformer.output_info)
        self.cond_generator = ConditionalVectorSampler(train_data, self.transformer.output_info)

        sides = [4, 8, 16, 24, 32]
        col_size_d = data_dim + self.cond_generator.n_options
        for i in sides:
            if i * i >= col_size_d:
                self.dside = i
                break

        sides = [4, 8, 16, 24, 32]
        col_size_g = data_dim
        for i in sides:
            if i * i >= col_size_g:
                self.gside = i
                break

        layers_G = determine_layers_gen(self.gside, self.random_dim + self.cond_generator.n_options, self.num_channels)
        layers_D = determine_layers_disc(self.dside, self.num_channels)
        self.generator = Generator(layers_G).to(self.device)
        discriminator = Discriminator(layers_D).to(self.device)

        optimizer_params = dict(lr=2e-4, betas=(0.5, 0.9), eps=1e-3, weight_decay=self.l2scale)
        optimizerG = Adam(self.generator.parameters(), **optimizer_params)
        optimizerD = Adam(discriminator.parameters(), **optimizer_params)

        st_ed = None
        classifier = None
        optimizerC = None
        if target_index is not None:
            st_ed = get_st_ed(target_index, self.transformer.output_info)
            classifier = Classifier(data_dim, self.class_dim, st_ed).to(self.device)
            optimizerC = optim.Adam(classifier.parameters(), **optimizer_params)

        self.generator.apply(weights_init)
        discriminator.apply(weights_init)

        self.Gtransformer = ImageTransformer(self.gside)
        self.Dtransformer = ImageTransformer(self.dside)

        return data_sampler, discriminator, optimizerG, optimizerD, classifier, optimizerC, st_ed

    def training_loop(self, train_data, steps_per_epoch, data_sampler, optimizerG, optimizerD, classifier, optimizerC,
                      st_ed, problem_type, discriminator):
        for i in tqdm(range(self.epochs)):
            for _ in range(steps_per_epoch):
                real_cat_d, real = self.train_discriminator(data_sampler, optimizerD, discriminator)
                noisez = self.train_generator(optimizerG, discriminator, real_cat_d)

                if problem_type:
                    self.train_classifier(optimizerC, classifier, st_ed, real)
                    self.train_generator_with_classifier(optimizerG, classifier, st_ed, noisez)

    def train_discriminator(self, data_sampler, optimizerD, discriminator):
        # discriminator training steps go here
        # sampling noise vectors using a standard normal distribution
        flag = False

        while flag is False:
            # sampling conditional vectors
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample_for_training(self.batch_size)
            c, m, col, opt = condvec
            c = torch.from_numpy(c).to(self.device)
            m = torch.from_numpy(m).to(self.device)
            # concatenating conditional vectors and converting resulting noise vectors into the image domain
            # to be fed to the generator as input
            noisez = torch.cat([noisez, c], dim=1)
            noisez = noisez.view(self.batch_size, self.random_dim + self.cond_generator.n_options, 1, 1)

            # sampling real data according to the conditional vectors and shuffling it before feeding to discriminator
            # to isolate conditional loss on generator
            perm = np.arange(self.batch_size)
            np.random.shuffle(perm)
            real, flag = data_sampler.sample_data(self.batch_size, col[perm], opt[perm])
            real = torch.from_numpy(real.astype('float32')).to(self.device)

        # storing shuffled ordering of the conditional vectors
        c_perm = c[perm]
        # generating synthetic data as an image
        fake = self.generator(noisez)
        # converting it into the tabular domain as per format of the transformed training data
        faket = self.Gtransformer.inverse_transform(fake)
        # applying final activation on the generated data (i.e., tanh for numeric and gumbel-softmax for categorical)
        fakeact = apply_activate(faket, self.transformer.output_info)

        # the generated data is then concatenated with the corresponding condition vectors
        fake_cat = torch.cat([fakeact, c], dim=1)
        # the real data is also similarly concatenated with corresponding conditional vectors
        real_cat = torch.cat([real, c_perm], dim=1)

        # transforming the real and synthetic data into the image domain for feeding it to the discriminator
        real_cat_d = self.Dtransformer.transform(real_cat)
        fake_cat_d = self.Dtransformer.transform(fake_cat)

        # executing the gradient update step for the discriminator
        optimizerD.zero_grad()
        # computing the probability of the discriminator to correctly classify real samples hence
        # y_real should ideally be close to 1
        y_real, _ = discriminator(real_cat_d)
        # computing the probability of the discriminator to correctly classify fake samples hence
        # y_fake should ideally be close to 0
        y_fake, _ = discriminator(fake_cat_d)
        # computing the loss to essentially maximize the log likelihood of correctly classifying
        # real and fake samples as log(D(x))+log(1−D(G(z)))
        # or equivalently minimizing the negative of log(D(x))+log(1−D(G(z))) as done below
        loss_d = (-(torch.log(y_real + 1e-4).mean()) - (torch.log(1. - y_fake + 1e-4).mean()))
        # accumulating gradients based on the loss
        loss_d.backward()
        # computing the backward step to update weights of the discriminator
        optimizerD.step()
        return real_cat_d, real

    def train_generator(self, optimizerG, discriminator, real_cat_d):
        # generator training steps go here
        # similarly sample noise vectors and conditional vectors
        noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
        condvec = self.cond_generator.sample_for_training(self.batch_size)
        c, m, col, opt = condvec
        c = torch.from_numpy(c).to(self.device)
        m = torch.from_numpy(m).to(self.device)
        noisez = torch.cat([noisez, c], dim=1)
        noisez = noisez.view(self.batch_size, self.random_dim + self.cond_generator.n_options, 1, 1)

        # executing the gradient update step for the generator
        optimizerG.zero_grad()

        # similarly generating synthetic data and applying final activation
        fake = self.generator(noisez)
        faket = self.Gtransformer.inverse_transform(fake)
        fakeact = apply_activate(faket, self.transformer.output_info)
        # concatenating conditional vectors and converting it to the image domain to be fed to the discriminator
        fake_cat = torch.cat([fakeact, c], dim=1)
        fake_cat = self.Dtransformer.transform(fake_cat)

        # computing the probability of the discriminator classifying fake samples as real
        # along with feature representations of fake data resulting from the penultimate layer
        y_fake, info_fake = discriminator(fake_cat)
        # extracting feature representation of real data from the penultimate layer of the discriminator
        _, info_real = discriminator(real_cat_d)
        # computing the conditional loss to ensure the generator generates data records with the chosen category as per
        # the conditional vector
        cross_entropy = cond_loss(faket, self.transformer.output_info, c, m)

        # computing the loss to train the generator where we want y_fake to be close to 1 to fool the discriminator
        # and cross_entropy to be close to 0 to ensure generator's output matches the conditional vector
        g = -(torch.log(y_fake + 1e-4).mean()) + cross_entropy
        # in order to backprop the gradient of separate losses w.r.t to the learnable weight of the network
        # independently
        # we may use retain_graph=True in backward() method in the first back-propagated loss
        # to maintain the computation graph to execute the second backward pass efficiently
        g.backward(retain_graph=True)
        # computing the information loss by comparing means and stds of real/fake feature representations extracted
        # from discriminator's penultimate layer
        loss_mean = torch.norm(torch.mean(info_fake.view(self.batch_size, -1), dim=0) - torch.mean(
            info_real.view(self.batch_size, -1), dim=0), 1)
        loss_std = torch.norm(torch.std(info_fake.view(self.batch_size, -1), dim=0) - torch.std(
            info_real.view(self.batch_size, -1), dim=0), 1)
        loss_info = loss_mean + loss_std
        # computing the finally accumulated gradients
        loss_info.backward()
        # executing the backward step to update the weights
        optimizerG.step()
        return noisez

    def train_classifier(self, optimizerC, classifier, st_ed, real):
        # classifier training steps go here
        self.c_loss = None
        # in case of binary classification, the binary cross entropy loss is used
        if (st_ed[1] - st_ed[0]) == 2:
            self.c_loss = BCELoss()
        # in case of multi-class classification, the standard cross entropy loss is used
        else:
            self.c_loss = CrossEntropyLoss()

        # updating the weights of the classifier
        optimizerC.zero_grad()
        # computing classifier's target column predictions on the real data along with returning
        # corresponding true labels
        real_pre, real_label = classifier(real)
        if (st_ed[1] - st_ed[0]) == 2:
            real_label = real_label.type_as(real_pre)
        # computing the loss to train the classifier so that it can perform well on the real data
        loss_cc = self.c_loss(real_pre, real_label)
        loss_cc.backward()
        optimizerC.step()

    def train_generator_with_classifier(self, optimizerG, classifier, st_ed, noisez):
        # updating the weights of the generator
        optimizerG.zero_grad()
        # generate synthetic data and apply the final activation
        fake = self.generator(noisez)
        faket = self.Gtransformer.inverse_transform(fake)
        fakeact = apply_activate(faket, self.transformer.output_info)
        # computing classifier's target column predictions on the fake data along with returning corresponding
        # true labels
        fake_pre, fake_label = classifier(fakeact)
        if (st_ed[1] - st_ed[0]) == 2:
            fake_label = fake_label.type_as(fake_pre)
        # computing the loss to train the generator to improve semantic integrity between target column and
        # rest of the data
        loss_cg = self.c_loss(fake_pre, fake_label)
        loss_cg.backward()
        optimizerG.step()

    def sample(self, n):

        # turning the generator into inference mode to effectively use running statistics in batch norm layers
        self.generator.eval()
        # column information associated with the transformer fit to the pre-processed training data
        output_info = self.transformer.output_info

        # generating synthetic data in batches accordingly to the total no. required
        steps = n // self.batch_size + 1
        data = []
        for _ in range(steps):
            # generating synthetic data using sampled noise and conditional vectors
            noisez = torch.randn(self.batch_size, self.random_dim, device=self.device)
            condvec = self.cond_generator.sample_for_generation(self.batch_size)
            c = condvec
            c = torch.from_numpy(c).to(self.device)
            noisez = torch.cat([noisez, c], dim=1)
            noisez = noisez.view(self.batch_size, self.random_dim + self.cond_generator.n_options, 1, 1)
            fake = self.generator(noisez)
            faket = self.Gtransformer.inverse_transform(fake)
            fakeact = apply_activate(faket, output_info)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)

        # applying the inverse transform and returning synthetic data in a similar form as the
        # original pre-processed training data
        result = self.transformer.inverse_transform(data)

        return result[0:n]
