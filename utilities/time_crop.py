import argparse

parser = argparse.ArgumentParser(description='Crops some contiguous times from a dataset')
parser.add_argument('t_i', type=int, help='initial time to include')
parser.add_argument('t_f', type=int, help='final time to include')
parser.add_argument('overwrite', default=False, help='overwrites file (default:False)')
parser.add_argument('file_to', default="", help="destination file")

parser.parse_args()



