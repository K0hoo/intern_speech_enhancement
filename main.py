
import argparse

import exec_real_img_norm


"""
When you want to test other situations, follow the below step.
1. Make a new file with the foramt file (exec_format.py)
2. If you want to use new model, make it on the model.py
3. Adjust some settings on the new exec_<>.py
4. Adjust soem argument through below parser and the file of main function.
"""
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root_folder', type=str, required=True)
    parser.add_argument('--model_parameter', type=str, required=True)
    parser.add_argument('--test_seen_result', type=str, required=True)
    parser.add_argument('--test_unseen_result', type=str, required=True)
    parser.add_argument('--output_train_log', type=str, required=True)
    parser.add_argument('--output_test_log', type=str, required=True)
    parser.add_argument('--train', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--fp16', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--clip', default=0.25, type=float, help='gradient clip threshold')
    parser.add_argument('--continue_epoch', type=int, default=0)
    args = parser.parse_args()

    # execute the main function of your own file!
    exec_real_img_norm.main(args)
