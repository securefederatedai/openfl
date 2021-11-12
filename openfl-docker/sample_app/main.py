import pynvml
from dep_module import func_
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('foo')
    args = parser.parse_args()

    assert args.foo == 'bar', "ARGUMENT bar WAS NOT READ"

    result = func_('Don')
    assert result == 'Hi, Don!'
    print('\n\n------------- IT WORKS -------------\n\n')