
import crm_complex.LSTM.set_lstm as set_lstm
import crm_complex.TCN.set_tcn as set_tcn

def main(args):

    setting_dict = {
        'LSTM': set_lstm,
        'TCN': set_tcn,
    }

    setting_dict[args.model].set(args)
