
import crm_complex_sisnr.TCN.set_tcn as set_tcn

def main(args):

    setting_dict = {
        'TCN': set_tcn
    }

    setting_dict[args.model].set(args)
    