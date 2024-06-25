
import os
import torch
from models import WaveEncoder, WaveDecoder
from utils import  adain
from torchvision.transforms import transforms
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--transfer_at_encoder', action='store_true',default='-e')
parser.add_argument('-d', '--transfer_at_decoder', action='store_true',default='-d')
parser.add_argument('-s', '--transfer_at_skip', action='store_true',default='-s')
parser.add_argument('--option_unpool', type=str, default='cat5')
parser.add_argument('--verbose', action='store_true',default="--verbose")
parser.add_argument('--model_path', type=str, default='./train_model', help='wave')
opt = parser.parse_args()
device = torch.device('cuda')
def WAVE(content,style):

      transfer_at = set()
      if opt.transfer_at_encoder:
          transfer_at.add('encoder')
      if opt.transfer_at_decoder:
          transfer_at.add('decoder')
      if opt.transfer_at_skip:
          transfer_at.add('skip')
  
      # The filenames of the content and style pair should match & intersection
  
      postfix = '_'.join(sorted(list(transfer_at)))
      wct = WCT(transfer_at=transfer_at, option_unpool=opt.option_unpool, device=device,
                        verbose=opt.verbose)
      with torch.no_grad():
          img=wct.transfer(content, style)
      return img


IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG',
    ]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class WCT:
    def __init__(self, model_path=opt.model_path, transfer_at=['encoder', 'skip', 'decoder'],
                 option_unpool='cat5', device='cuda', verbose=False):

        self.transfer_at = set(transfer_at)
        assert not (self.transfer_at - set(['encoder', 'decoder', 'skip'])), 'invalid transfer_at: {}'.format(
            transfer_at)
        assert self.transfer_at, 'empty transfer_at'

        self.device = torch.device(device)
        self.verbose = verbose
        self.encoder = WaveEncoder(option_unpool).to(self.device)
        self.decoder = WaveDecoder(option_unpool).to(self.device)
        self.encoder.load_state_dict(
            torch.load(os.path.join(model_path, 'wave_encoder_{}_l4.pth'.format(option_unpool)),
                       map_location=lambda storage, loc: storage))
        self.decoder.load_state_dict(
            torch.load(os.path.join(model_path, 'wave_decoder_{}_l4.pth'.format(option_unpool)),
                       map_location=lambda storage, loc: storage))

    def print_(self, msg):
        if self.verbose:
            print(msg)

    def encode(self, x, skips, level):
        return self.encoder.encode(x, skips, level)

    def decode(self, x, skips, level):
        return self.decoder.decode(x, skips, level)

    def get_all_feature(self, x):
        skips = {}
        feats = {'encoder': {}, 'decoder': {}}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)
            if 'encoder' in self.transfer_at:
                feats['encoder'][level] = x

        if 'encoder' not in self.transfer_at:
            feats['decoder'][4] = x
        for level in [4, 3, 2]:
            x = self.decode(x, skips, level)
            if 'decoder' in self.transfer_at:
                feats['decoder'][level - 1] = x
        return feats, skips

    def transfer(self, content, style, alpha=1):
        content_feat, content_skips = content, {}
        style_feats, style_skips = self.get_all_feature(style)

        wct_enc_level = [1, 2, 3, 4]
        wct_dec_level = [1, 2, 3, 4]
        wct_skip_level = ['pool1', 'pool2', 'pool3']

        for level in [1, 2, 3, 4]:
            content_feat = self.encode(content_feat, content_skips, level)
            if 'encoder' in self.transfer_at and level in wct_enc_level:
                content_feat = adain(content_feat, style_feats['encoder'][level])
        if 'skip' in self.transfer_at:
            for skip_level in wct_skip_level:
                for component in [0, 1, 2]:  # component: [LH, HL, HH]
                    content_skips[skip_level][component] = adain(content_skips[skip_level][component],
                                                                 style_skips[skip_level][component])
        for level in [4, 3, 2, 1]:
            if 'decoder' in self.transfer_at and level in style_feats['decoder'] and level in wct_dec_level:
                content_feat = adain(content_feat, style_feats['decoder'][level])

            content_feat = self.decode(content_feat, content_skips, level)
        return content_feat

'''
def get_all_transfer():
    ret = []
    for e in ['encoder', None]:
        for d in ['decoder', None]:
            for s in ['skip', None]:
                _ret = set([e, d, s]) & set(['encoder', 'decoder', 'skip'])
                if _ret:
                    ret.append(_ret)
    return ret'''