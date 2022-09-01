
import magnitude.LSTM.set_lstm as set_lstm
import magnitude.TCN.set_tcn as set_tcn

def main(args):
    setting_dict = {
        'LSTM': set_lstm.set,
        'TCN': set_tcn.set
    }

    setting_dict[args.model](args)