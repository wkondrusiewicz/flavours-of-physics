import pandas as pd
import evaluation


def check_combined_ks_and_cvm(w1, w2, mod_1, mod_2, check_agreement, check_correlation, flag=False):
    print('Checking KS and CVM for combined models with weights {} and {}\n'.format(w1, w2))
    assert w1 + w2 == 1, "Wieghts must sum up to 1"
    agr_1 = mod_1.agreement_probs
    agr_2 = mod_2.agreement_probs
    cor_1 = mod_1.correlation_probs
    cor_2 = mod_2.correlation_probs
    agr = w1 * agr_1 + w2 * agr_2
    cor = w1 * cor_1 + w2 * cor_2
    ks = evaluation.compute_ks(
        agr[check_agreement['signal'].values == 0],
        agr[check_agreement['signal'].values == 1],
        check_agreement[check_agreement['signal'] == 0]['weight'].values,
        check_agreement[check_agreement['signal'] == 1]['weight'].values)
    cvm = evaluation.compute_cvm(cor, check_correlation['mass'])
    print('KS metric = {}. Is it smaller than 0.09? {}'.format(ks, ks < 0.09))
    print('CVM metric = {}. Is it smaller than 0.002? {}\n'.format(cvm, cvm < 0.002))
    if flag == True:
        return ks, cvm


def save_combined_output(filename, data_to_predict, w1, w2, mod_1, mod_2):
    assert w1 + w2 == 1, "Wieghts must sum up to 1"
    print('Saving combined models with weights {} and {} in file '.format(
        w1, w2) + filename + '\n')
    pred_1 = mod_1.predicted
    pred_2 = mod_2.predicted
    result = w1 * pred_1 + w2 * pred_2
    res = pd.DataFrame({'id': data_to_predict.index})
    res['prediction'] = result
    res.to_csv(filename, index=False, header=True, sep=',')
