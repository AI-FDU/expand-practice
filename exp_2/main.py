import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tqdm import *
from PIL import Image

from flickr.data import get_loader_single, get_transform, collate_fn, encode_data, i2t, t2i
from matplotlib import pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = 'data/'
IMAGE_FOLDER = 'data/Flicker8k_Dataset'
VOCAB_PATH = 'vocab'
PRETRAINED = 'saves'
with open('{}/flickr_precomp_vocab.pkl'.format(VOCAB_PATH), 'rb') as f:
    vocab = pickle.load(f)

batch_size = 32


def get_loaders(root, vocab, batch_size):
    transform = get_transform('train')

    train_loader = get_loader_single(root,
                                     'train',
                                     vocab,
                                     transform,
                                     batch_size=batch_size,
                                     shuffle=True,
                                     collate_fn=collate_fn)

    transform = get_transform('dev')

    val_loader = get_loader_single(root,
                                   'dev',
                                   vocab,
                                   transform,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   collate_fn=collate_fn)
    transform = get_transform('test')

    return train_loader, val_loader


train_loader, val_loader = get_loaders(ROOT, vocab, batch_size)


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        # Load a pre-trained model

        self.cnn = models.__dict__['vgg19'](pretrained=True)
        self.cnn.features = nn.DataParallel(self.cnn.features)
        self.cnn.cuda()
        # Replace the last fully connected layer of CNN with a new one
        self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features,
                            embed_size)
        self.cnn.classifier = nn.Sequential(
            *list(self.cnn.classifier.children())[:-1])

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)

        # normalization in the image embedding space
        features = l2norm(features)

        # linear projection to the joint embedding space
        features = self.fc(features)

        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)
        if self.use_abs:
            features = torch.abs(features)

        # take the absolute value of the embedding (used in order embeddings)
        return features


class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, use_abs=False):
        super(EncoderText, self).__init__()

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.embed_size = embed_size
        self.use_abs = use_abs
        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size) - 1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)

        return out


class PairwiseRankingLoss(torch.nn.Module):

    def __init__(self, margin=1.0):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s):
        margin = self.margin
        # compute image-sentence score matrix
        scores = torch.mm(im, s.transpose(1, 0))
        diagonal = scores.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(
            Variable(torch.zeros(scores.size()[0],
                                 scores.size()[1]).cuda()),
            (margin - diagonal).expand_as(scores) + scores)
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = torch.max(
            Variable(torch.zeros(scores.size()[0],
                                 scores.size()[1]).cuda()),
            (margin - diagonal).expand_as(scores).transpose(1, 0) + scores)

        for i in range(scores.size()[0]):
            cost_s[i, i] = 0
            cost_im[i, i] = 0

        return cost_s.sum() + cost_im.sum()


batch_size = 32
vocab_size = len(vocab)
print('Dictionary size: ' + str(vocab_size))
embed_size = 1024
img_dim = 4096
word_dim = 300
num_epochs = 1
img_enc = EncoderImageFull(embed_size).to(device)
txt_enc = EncoderText(vocab_size, word_dim, embed_size).to(device)
params = list(txt_enc.parameters())
params += list(img_enc.fc.parameters())
criterion = PairwiseRankingLoss(margin=0.2)
optimizer = torch.optim.Adam(params, lr=0.0002)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = 0.0002 * (0.1**(epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def validate(val_loader, txt_enc, img_enc):
    img_embs, cap_embs = encode_data(txt_enc, img_enc, val_loader)
    r1, r5, r10, medr = i2t(img_embs, cap_embs)
    r1i, r5i, r10i, medri = t2i(img_embs, cap_embs)
    score = r1 + r5 + r10 + r1i + r5i + r10i
    return r1


start_epoch = 0
for epoch in range(start_epoch, num_epochs):
    adjust_learning_rate(optimizer, epoch)
    for i, batch in enumerate(tqdm(train_loader)):
        images, captions, lengths, ids = batch
        images = images.to(device)
        captions = captions.to(device)
        img_emb = img_enc(images)
        cap_emb = txt_enc(captions, lengths)
        loss = criterion(img_emb, cap_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    rsum = validate(val_loader, txt_enc, img_enc)
    print('Epochs: [%d]/[%d] AvgScore: %.2f Loss: %.2f' %
          (epoch, num_epochs, rsum, loss.item()))

    if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
        torch.save(
            {
                'text_encoder': txt_enc.state_dict(),
                'image_encoder': img_enc.state_dict(),
                'epoch': epoch,
                'optimizer': optimizer.state_dict(),
            }, f'saves/model_epoch_{epoch+1}.pth')
        print(f"Full model saved at epoch {epoch+1}")


def load_image(file_name):
    image = Image.open(file_name)
    return image


def load_checkpoint(savefile):
    if os.path.exists(savefile):
        checkpoint = torch.load(savefile)
        return checkpoint
    else:
        print('No checkpoints available')


def get_captions():
    with open('{}/f8k_train_caps.txt'.format(ROOT), 'r') as f:
        lines = f.readlines()
    captions = [line.strip() for line in lines]
    return captions


def text_retrieval(image_embedding, cap_embs, captions):
    scores = np.dot(image_embedding, cap_embs.T).flatten()
    sorted_args = np.argsort(scores)[::-1]
    sentences = [captions[a] for a in sorted_args[:10]]
    return sentences


savefile = f'saves/model_epoch_{num_epochs}.pth'
img_enc.eval()
txt_enc.eval()
checkpoint = load_checkpoint(savefile)
img_enc.load_state_dict(checkpoint['image_encoder'])
txt_enc.load_state_dict(checkpoint['text_encoder'])
img_embs, cap_embs = encode_data(txt_enc, img_enc, train_loader)

query_image = load_image('{}/106490881_5a2dd9b7bd.jpg'.format(IMAGE_FOLDER))
transform = get_transform('dev')
query_image = transform(query_image)
query_image_embedding = img_enc(query_image.unsqueeze(0)).data.cpu().numpy()
captions = get_captions()
ret = text_retrieval(query_image_embedding, cap_embs, captions)
for each_ret in ret:
    print(each_ret + '\n')
