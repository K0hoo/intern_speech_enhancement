
import complex.LSTM.set_lstm as set_lstm

def main(args):

    setting_dict = {
        'LSTM': set_lstm.set,
    }

    setting_dict[args.model](args)