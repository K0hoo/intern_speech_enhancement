
import argparse

from crm_complex import main as crm_complex
from wiener_mag import main as wiener_mag
from wiener_log_mag import main as wiener_log_mag

"""
When you want to test other situations, follow the below step.
1. Make a new file with the foramt file (exec_format.py)
2. If you want to use new model, make it on the model.py
3. Adjust some settings on the new exec_<>.py
4. Adjust soem argument through below parser and the file of main function.
"""

target_dict = {
    'wiener_mag': wiener_mag,
    'wiener_log_mag': wiener_log_mag,
    'crm_complex': crm_complex
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, required=True)
    parser.add_argument('--data_root_folder', type=str, required=True)
    parser.add_argument('--target_type', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--clip', default=0.25, type=float, help='gradient clip threshold')
    args = parser.parse_args()
    

    # execute the main function of your own file!
    target_dict[args.target_type].main(args)
