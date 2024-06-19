import argparse

from model import TextCorrector

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_config', type=str, default='./config/data.yaml')
    parser.add_argument('--model_config', type=str, default='./config/model.yaml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--detector_path', type=str, default='output/ckpt/detector/4.pth')
    parser.add_argument('--corrector_path', type=str, default='output/ckpt/corrector/4.pth')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    model = TextCorrector(**)