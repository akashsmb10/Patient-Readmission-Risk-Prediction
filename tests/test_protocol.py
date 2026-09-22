import numpy as np
import pandas as pd
from run import FEATURES, NUMERIC, CATEGORICAL, make_splits, prepare, preprocessor, choose_threshold


def fixture_frame(n=400):
    df=pd.DataFrame({c:np.arange(n)%5+1 for c in NUMERIC})
    for c in CATEGORICAL: df[c]='None' if c=='A1Cresult' else 'known'
    df['discharge_disposition_id']=1
    df['encounter_id']=np.arange(n)
    df['patient_nbr']=np.arange(n)//2
    df['readmitted']=np.where(np.arange(n)%3==0,'<30','NO')
    return df


def test_group_splits_are_disjoint_exhaustive_reproducible():
    df=prepare(fixture_frame()); splits=make_splits(df)
    assert sum(map(len,splits.values()))==len(df)
    for name,indices in splits.items():
        assert np.array_equal(indices,make_splits(df)[name])
        for other,other_indices in splits.items():
            if name!=other:
                assert set(df.iloc[indices].patient_nbr).isdisjoint(df.iloc[other_indices].patient_nbr)


def test_target_and_discharge_exclusions():
    df=fixture_frame(); df.loc[0,'discharge_disposition_id']=11
    df.loc[1,'discharge_disposition_id']=13
    cleaned=prepare(df)
    assert len(cleaned)==len(df)-2
    assert (cleaned.target==(cleaned.readmitted=='<30').astype(int)).all()
    assert set(FEATURES).isdisjoint({'patient_nbr','encounter_id','readmitted','target','race','gender'})
    assert cleaned.A1Cresult.eq('None').all()


def test_preprocessing_fits_only_train_and_accepts_unseen_categories():
    train=prepare(fixture_frame(20))[FEATURES]
    train['num_medications']=np.arange(20)
    future=train.iloc[:2].copy(); future['num_medications']=9999; future['diag_1']='unseen'
    transformer=preprocessor().fit(train)
    median=transformer.named_transformers_['numeric'].named_steps['impute'].statistics_[NUMERIC.index('num_medications')]
    assert median==9.5
    before=transformer.named_transformers_['numeric'].named_steps['scale'].mean_.copy()
    assert transformer.transform(future).shape[0]==2
    assert np.array_equal(before,transformer.named_transformers_['numeric'].named_steps['scale'].mean_)


def test_threshold_uses_supplied_validation_labels():
    y=np.array([0,0,1,1]); p=np.array([0.01,0.05,0.4,0.8])
    threshold=choose_threshold(y,p)
    assert 0.05 < threshold <= 0.4
