import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_new_features(train_path='../data/training.csv', check_agreement_path='../data/check_agreement.csv', check_correlation_path='../data/check_correlation.csv', test_path='../data/test.csv', no_test=False):
    train = pd.read_csv(train_path, index_col='id')
    check_agreement = pd.read_csv(check_agreement_path, index_col='id')
    check_correlation = pd.read_csv(check_correlation_path, index_col='id')
    if no_test == False:
        test = pd.read_csv(test_path, index_col='id')
    else:
        test = None

    # new variables
    c = 29.9792458  # speed of light in cm/ns
    m_mu = 105.6583715  # Muon mass (in MeV/c)
    m_tau = 1776.82    # Tau mass (in MeV/c)
    data_frames = [train, check_agreement, check_correlation, test]
    data_frames = [e for e in data_frames if e is not None]
    for df in data_frames:
        df['p0_pz'] = (df.p0_p**2 - df.p0_pt**2)**0.5
        df['p1_pz'] = (df.p1_p**2 - df.p1_pt**2)**0.5
        df['p2_pz'] = (df.p2_p**2 - df.p2_pt**2)**0.5
        df['pz'] = df.p0_pz + df.p1_pz + df.p2_pz
        df['p'] = np.sqrt(df.pt**2 + df.pz**2)
        df['E0'] = np.sqrt(m_mu**2 + df.p0_p**2)
        df['E1'] = np.sqrt(m_mu**2 + df.p1_p**2)
        df['E2'] = np.sqrt(m_mu**2 + df.p2_p**2)
        df['E'] = np.sqrt(m_mu**2 + df.p0_p**2) + np.sqrt(m_mu
                                                          ** 2 + df.p1_p**2) + np.sqrt(m_mu**2 + df.p2_p**2)
        df['E0_ratio'] = df.E0 / df.E
        df['E1_ratio'] = df.E1 / df.E
        df['E2_ratio'] = df.E2 / df.E
        df['p0_pt_ratio'] = df.p0_pt / df.pt
        df['p1_pt_ratio'] = df.p1_pt / df.pt
        df['p2_pt_ratio'] = df.p2_pt / df.pt
        df['eta01'] = df.p0_eta - df.p1_eta
        df['eta12'] = df.p1_eta - df.p2_eta
        df['eta20'] = df.p2_eta - df.p0_eta
        df['coll_pt'] = (df.p0_pt + df.p1_pt + df.p2_pt) / df.pt
        df['CDF_sum'] = df.CDF3 + df.CDF3 + df.CDF3
        df['isolation_sum'] = df.isolationa + df.isolationb + \
            df.isolationc + df.isolationd + df.isolatione + df.isolationf
        df['DOCA_sum'] = df.DOCAone + df.DOCAtwo + df.DOCAthree
        df['IsoBDT_sum'] = df.p0_IsoBDT + df.p1_IsoBDT + df.p2_IsoBDT
        df['IP_sum'] = df.p0_IP + df.p1_IP + df.p2_IP
        df['IPSig_sum'] = df.p0_IPSig + df.p1_IPSig + df.p2_IPSig
        df['mass_by_E'] = np.sqrt(df.E ** 2 - df.p ** 2)
        df['gamma'] = df.E / df.mass_by_E
        df['beta'] = np.sqrt(df.gamma**2 - 1) / df.gamma
        df['track_Chi2Dof_sum'] = np.sqrt(
            (df.p0_track_Chi2Dof - 1)**2 + (df.p1_track_Chi2Dof - 1)**2 + (df.p2_track_Chi2Dof - 1)**2)
        df['velocity'] = df.FlightDistance / df.LifeTime

    train = train.dropna()
    train = train[train['min_ANNmuon'] > 0.4]
    check_agreement = check_agreement.dropna()
    check_correlation = check_correlation.dropna()
    return train, check_agreement, check_correlation, test


def variables_list():
    var_geo = ['LifeTime', 'dira', 'FlightDistance', 'FlightDistanceError', 'IP', 'IPSig',  'VertexChi2', 'pt', 'DOCAone', 'DOCAtwo', 'DOCAthree', 'IP_p0p2', 'IP_p1p2', 'isolationa', 'isolationb', 'isolationc', 'isolationd', 'isolatione', 'isolationf', 'iso', 'CDF1', 'CDF2', 'CDF3',
               'ISO_SumBDT', 'p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT', 'p0_track_Chi2Dof', 'p1_track_Chi2Dof', 'p2_track_Chi2Dof', 'p0_IP', 'p1_IP', 'p2_IP', 'p0_IPSig', 'p1_IPSig', 'p2_IPSig', 'E', 'CDF_sum', 'isolation_sum', 'DOCA_sum', 'IsoBDT_sum', 'IP_sum', 'IPSig_sum', 'track_Chi2Dof_sum']
    var_kin = ['E', 'dira', 'pt', 'p0_pt', 'p1_pt', 'p2_pt', 'p', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta', 'pz', 'p0_pz', 'p1_pz', 'p2_pz', 'eta01',
               'eta12', 'eta20', 'coll_pt', 'E0', 'E1', 'E2', 'E0_ratio', 'E1_ratio', 'E2_ratio', 'p0_pt_ratio', 'p1_pt_ratio', 'p2_pt_ratio', 'gamma', 'beta', 'mass_by_E']
    return var_kin, var_geo
