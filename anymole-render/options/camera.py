import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Camera options')
    parser.add_argument("--znear", type=float, default=0.1, help="Near clipping plane")
    parser.add_argument("--zfar", type=float, default=100.0, help="Far clipping plane")
    parser.add_argument("--aspect_ratio", type=float, default=1.0, help="Aspect ratio")
    parser.add_argument("--fov", type=float, default=45.0, help="Field of view")
    return parser.parse_args(args=[])