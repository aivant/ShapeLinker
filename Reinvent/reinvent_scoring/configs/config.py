import argparse
import json
import os
from pathlib import Path


DEFAULT_BASE_CONFIG_PATH = (Path(__file__).parent / 'test_config.json').resolve()

parser = argparse.ArgumentParser(description='Reinvent Scoring configuration parser')
parser.add_argument(
    '--base_config', type=str, default=DEFAULT_BASE_CONFIG_PATH,
    help='Path to basic configuration for Reinvent Scoring environment.'
)


def read_json_file(path):
    with open(path) as f:
        json_input = f.read().replace('\r', '').replace('\n', '')
    try:
        return json.loads(json_input)
    except (ValueError, KeyError, TypeError) as e:
        print(f"JSON format error in file ${path}: \n ${e}")


args, _ = parser.parse_known_args()

reinvent_scoring_config = read_json_file(args.base_config)

for key, value in reinvent_scoring_config['ENVIRONMENTAL_VARIABLES'].items():
    os.environ[key] = value
