import argparse
import os, warnings
from omegaconf import OmegaConf

from configuration import CFG
import trainer.train_loop as train_loop
from utils.helper import check_library, all_type_seed
from utils.util import sync_config
from dataset_class.preprocessing import add_target_token, add_anchor_token
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["LRU_CACHE_CAPACITY"] = "1"

check_library(True)
all_type_seed(CFG, True)


def main(config_path: str, cfg: CFG) -> None:
    target_token, anchor_token = ' [TAR] ', ' [ANC] '
    sync_config(OmegaConf.load(config_path))  # load json config
    add_target_token(cfg, target_token), add_anchor_token(cfg, anchor_token)
    # cfg = OmegaConf.structured(CFG)
    # OmegaConf.merge(cfg)  # merge with cli_options
    getattr(train_loop, cfg.loop)(cfg)  # init object


if __name__ == '__main__':
    # args = argparse.ArgumentParser(description='PyTorch Template')
    # args.add_argument('-c', '--config', default=None, type=str,
    #                   help='config file path (default: None)')
    # args.add_argument('-r', '--resume', default=None, type=str,
    #                   help='path to latest checkpoint (default: None)')
    # args.add_argument('-d', '--device', default=None, type=str,
    #                   help='indices of GPUs to enable (default: all)')
    #
    # # custom cli options to modify configuration from default values given in json file.
    # CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    # options = [
    #     CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
    #     CustomArgs(['--bs', '--batch_size'], type=int, target='dataset_class;args;batch_size')
    # ]
    # cli_config = ConfigParser.from_args(args, options)
    main('one2one_train_cfg.json', CFG)

