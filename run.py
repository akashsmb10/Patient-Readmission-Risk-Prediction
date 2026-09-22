"""Reproducible discharge-time readmission benchmark; not a clinical tool."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import urllib.request
import zipfile
import warnings
import joblib
import numpy as np
import pandas as pd
import sklearn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
    precision_score, recall_score, fbeta_score, confusion_matrix, precision_recall_curve,
    roc_curve)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

ROOT=Path(__file__).resolve().parent
URL='https://archive.ics.uci.edu/static/public/296/diabetes%2B130-us%2Bhospitals%2Bfor%2Byears%2B1999-2008.zip'
SEED=360
NUMERIC=['time_in_hospital','num_lab_procedures','num_procedures','num_medications',
         'number_outpatient','number_emergency','number_inpatient','number_diagnoses']
CATEGORICAL=['age','admission_type_id','discharge_disposition_id','admission_source_id',
             'diag_1','diag_2','diag_3','max_glu_serum','A1Cresult','metformin','insulin',
             'change','diabetesMed']
FEATURES=NUMERIC+CATEGORICAL
EXCLUDED_DISPOSITIONS={11,13,14,19,20,21}


def load_data():
    directory=ROOT/'data/raw'; directory.mkdir(parents=True,exist_ok=True)
    archive=directory/'uci_diabetes.zip'
    if not archive.exists():
        temporary=archive.with_suffix('.download')
        urllib.request.urlretrieve(URL,temporary)
        with zipfile.ZipFile(temporary) as z: assert 'diabetic_data.csv' in z.namelist()
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as z:
        raw=z.read('diabetic_data.csv')
        (directory/'IDS_mapping.csv').write_bytes(z.read('IDS_mapping.csv'))
        # 'None' is a meaningful lab category (not measured); only '?' means missing.
        df=pd.read_csv(z.open('diabetic_data.csv'),keep_default_na=False,na_values=['?'])
    return df,hashlib.sha256(raw).hexdigest()


def prepare(df):
    assert df.encounter_id.is_unique, 'Duplicate encounters'
    assert df[['encounter_id','patient_nbr','readmitted']].notna().all().all()
    assert set(df.readmitted.unique())<= {'NO','>30','<30'}, 'Unknown target'
    cohort=df[~df.discharge_disposition_id.isin(EXCLUDED_DISPOSITIONS)].copy().reset_index(drop=True)
    for c in CATEGORICAL:
        cohort[c]=cohort[c].map(lambda v: str(v) if pd.notna(v) else np.nan)
    for c in NUMERIC:
        cohort[c]=pd.to_numeric(cohort[c],errors='raise')
        assert (cohort[c].dropna()>=0).all(), f'Invalid {c}'
    cohort['target']=(cohort.readmitted=='<30').astype(int)
    return cohort


def make_splits(df):
    # Approximate 60/15/10/15 proportions by patient, not by encounter.
    def split(indices,fraction,seed):
        a,b=next(GroupShuffleSplit(n_splits=1,test_size=fraction,random_state=seed)
                 .split(indices,groups=df.iloc[indices].patient_nbr))
        return indices[a],indices[b]
    remaining,test=split(np.arange(len(df)),0.15,SEED)
    remaining,validation=split(remaining,0.10/0.85,SEED+1)
    train,calibration=split(remaining,0.15/0.75,SEED+2)
    splits=dict(train=train,calibration=calibration,validation=validation,test=test)
    names=list(splits)
    assert len(np.unique(np.concatenate(list(splits.values()))))==len(df)
    for i,a in enumerate(names):
        assert df.iloc[splits[a]].target.nunique()==2
        for b in names[i+1:]:
            assert set(df.iloc[splits[a]].patient_nbr).isdisjoint(df.iloc[splits[b]].patient_nbr)
    return splits


def preprocessor():
    return ColumnTransformer([
        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),
                             ('scale',StandardScaler())]),NUMERIC),
        ('categorical',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='Missing')),
            ('encode',OneHotEncoder(handle_unknown='ignore',min_frequency=30,sparse_output=False))]),CATEGORICAL)
    ])


def log_odds(probabilities):
    p=np.clip(probabilities,1e-6,1-1e-6)
    return np.log(p/(1-p)).reshape(-1,1)


def calibrated_probability(model,calibrator,x):
    return calibrator.predict_proba(log_odds(model.predict_proba(x)[:,1]))[:,1]


def metrics(y,p,threshold):
    predicted=(p>=threshold).astype(int)
    return {'average_precision':float(average_precision_score(y,p)),
        'roc_auc':float(roc_auc_score(y,p)),'brier_score':float(brier_score_loss(y,p)),
        'precision':float(precision_score(y,predicted,zero_division=0)),
        'recall':float(recall_score(y,predicted,zero_division=0)),
        'f1':float(__import__('sklearn').metrics.f1_score(y,predicted,zero_division=0)),
        'f2':float(fbeta_score(y,predicted,beta=2,zero_division=0)),
        'flagged_fraction':float(predicted.mean()),'threshold':float(threshold)}


def choose_threshold(y,p):
    thresholds=np.linspace(0.02,0.60,117)
    scores=[fbeta_score(y,p>=t,beta=2,zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def cluster_intervals(y,p,groups,repeats=100):
    rng=np.random.default_rng(SEED)
    _,inverse=np.unique(groups,return_inverse=True)
    size=inverse.max()+1
    statistics=[]
    for _ in range(repeats):
        counts=np.bincount(rng.integers(0,size,size),minlength=size)[inverse]
        statistics.append([average_precision_score(y,p,sample_weight=counts),
            roc_auc_score(y,p,sample_weight=counts),brier_score_loss(y,p,sample_weight=counts)])
    bounds=np.quantile(statistics,[0.025,0.975],axis=0)
    return {name:{'low':float(bounds[0,i]),'high':float(bounds[1,i])}
        for i,name in enumerate(['average_precision','roc_auc','brier_score'])}


def make_plots(y,probs,selected,threshold,output):
    fig,axes=plt.subplots(2,2,figsize=(12,9))
    for name,p in probs.items():
        precision,recall,_=precision_recall_curve(y,p)
        axes[0,0].plot(recall,precision,label=name)
        fpr,tpr,_=roc_curve(y,p); axes[0,1].plot(fpr,tpr,label=name)
        actual,predicted=calibration_curve(y,p,n_bins=10,strategy='quantile')
        axes[1,0].plot(predicted,actual,'o-',label=name)
    axes[0,0].axhline(np.mean(y),color='gray',ls='--',label='Test prevalence')
    axes[0,0].set(xlabel='Recall',ylabel='Precision',title='Held-out precision–recall')
    axes[0,1].plot([0,1],[0,1],'k--'); axes[0,1].set(xlabel='False positive rate',ylabel='True positive rate',title='Held-out ROC')
    axes[1,0].plot([0,1],[0,1],'k--'); axes[1,0].set(xlabel='Mean predicted risk',ylabel='Observed rate',title='Calibration (quantile bins)')
    matrix=confusion_matrix(y,probs[selected]>=threshold)
    axes[1,1].imshow(matrix,cmap='Blues')
    for (i,j),v in np.ndenumerate(matrix): axes[1,1].text(j,i,str(v),ha='center',va='center')
    axes[1,1].set(xticks=[0,1],yticks=[0,1],xlabel='Predicted label',ylabel='Observed label',title=f'{selected}: validation-selected threshold {threshold:.3f}')
    for ax in axes.flat:
        if ax is not axes[1,1]: ax.legend(fontsize=8)
    fig.suptitle('Readmission benchmark — research use only',fontsize=15)
    fig.tight_layout(); fig.savefig(output/'evaluation.png',dpi=160); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--bootstrap',type=int,default=100)
    args=parser.parse_args()
    output=ROOT/'outputs'; output.mkdir(exist_ok=True)
    raw,checksum=load_data(); df=prepare(raw); splits=make_splits(df)
    audit={'source_rows':len(raw),'eligible_rows':len(df),'excluded_rows':len(raw)-len(df),
        'exclusion_dispositions':sorted(EXCLUDED_DISPOSITIONS),'source_csv_sha256':checksum,
        'patient_overlap':0,'seed':SEED,'prediction_time':'At discharge',
        'target':'Recorded readmission <30 days versus NO or >30',
        'features':FEATURES,'splits':{k:{'rows':len(v),'patients':int(df.iloc[v].patient_nbr.nunique()),
            'positive_rate':float(df.iloc[v].target.mean())} for k,v in splits.items()},
        'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,
                    'scikit_learn':sklearn.__version__,'joblib':joblib.__version__}}
    pd.DataFrame({'feature':raw.columns,'missing_fraction':[raw[c].isna().mean() for c in raw]}).to_csv(output/'data_quality.csv',index=False)
    manifest=pd.concat([df.iloc[v][['encounter_id','patient_nbr']].assign(split=k) for k,v in splits.items()])
    manifest.to_csv(output/'split_manifest.csv',index=False)
    data={k:(df.iloc[v][FEATURES],df.iloc[v].target.to_numpy()) for k,v in splits.items()}
    xt,yt=data['train']; xc,yc=data['calibration']; xv,yv=data['validation']; xe,ye=data['test']
    # Cap fitting only to keep this educational portfolio run practical on a laptop.
    # The cap is drawn from the already patient-disjoint training partition.
    fit_rows=min(12000,len(xt)); xt_fit,yt_fit=xt.iloc[:fit_rows],yt[:fit_rows]
    estimators={'dummy_prevalence':DummyClassifier(strategy='prior',random_state=SEED),
        'logistic_regression':LogisticRegression(max_iter=1500,solver='liblinear',random_state=SEED),
        'random_forest':RandomForestClassifier(n_estimators=40,min_samples_leaf=30,max_features='sqrt',
                                               n_jobs=1,random_state=SEED),
        'gradient_boosting':GradientBoostingClassifier(n_estimators=40,min_samples_leaf=40,
                                                       learning_rate=0.05,max_depth=2,random_state=SEED)}
    models={}; calibrators={}; selection=[]
    for name,estimator in estimators.items():
        print(f'Training {name} on {len(yt_fit):,} encounters',flush=True)
        model=Pipeline([('preprocess',preprocessor()),('model',estimator)])
        model.fit(xt_fit,yt_fit)
        if name == 'dummy_prevalence':
            calibration=None
            p=model.predict_proba(xv)[:,1]
        else:
            calibration=LogisticRegression(C=1e6,solver='lbfgs',random_state=SEED)
            calibration.fit(log_odds(model.predict_proba(xc)[:,1]),yc)
            assert calibration.coef_[0,0]>0, 'Calibration unexpectedly reverses model ranking'
            p=calibrated_probability(model,calibration,xv)
        threshold=choose_threshold(yv,p)
        selection.append({'model':name,**metrics(yv,p,threshold)})
        models[name]=model; calibrators[name]=calibration
    selection_df=pd.DataFrame(selection).sort_values('average_precision',ascending=False)
    selection_df.to_csv(output/'validation_comparison.csv',index=False)
    selected=selection_df[selection_df.model != 'dummy_prevalence'].iloc[0]['model']; threshold=float(selection_df.set_index('model').loc[selected,'threshold'])
    print(f'Selected on validation only: {selected}; threshold={threshold:.3f}',flush=True)
    # Freeze selection before evaluating test outcomes. Both candidate test results are reported transparently.
    probs={'prevalence_baseline':np.full(len(ye),yt.mean())}
    evaluation=[]
    for name in models:
        raw_p=models[name].predict_proba(xe)[:,1]
        p=raw_p if calibrators[name] is None else calibrated_probability(models[name],calibrators[name],xe); probs[name]=p
        t=float(selection_df.set_index('model').loc[name,'threshold'])
        evaluation.append({'model':name,**metrics(ye,p,t),'uncalibrated_brier':float(brier_score_loss(ye,raw_p))})
    evaluation.append({'model':'prevalence_baseline',**metrics(ye,probs['prevalence_baseline'],0.5),
                       'uncalibrated_brier':float(brier_score_loss(ye,probs['prevalence_baseline']))})
    pd.DataFrame(evaluation).to_csv(output/'test_metrics.csv',index=False)
    p=probs[selected]
    audit.update({'selected_model':selected,'threshold_selection':'Maximum validation F2 on a fixed grid; exploratory, not clinical utility',
        'selected_threshold':threshold,'bootstrap_repeats':args.bootstrap,
        'training_fit_rows':fit_rows,
        'test_cluster_bootstrap_95pct':cluster_intervals(ye,p,df.iloc[splits['test']].patient_nbr,args.bootstrap)})
    predictions=df.iloc[splits['test']][['encounter_id','patient_nbr','race','gender','age']].copy()
    predictions['target']=ye; predictions['probability']=p
    predictions.to_csv(output/'test_predictions.csv',index=False)
    group_rows=[]
    for col in ['race','gender','age']:
        for value,group in predictions.groupby(col,dropna=False):
            if len(group)<100 or group.target.nunique()<2: continue
            group_rows.append({'attribute':col,'group':value,'n':len(group),'prevalence':group.target.mean(),
                **metrics(group.target,group.probability,threshold)})
    pd.DataFrame(group_rows).to_csv(output/'subgroup_audit.csv',index=False)
    print('Computing validation permutation importance',flush=True)
    importance=permutation_importance(models[selected],xv,yv,scoring='average_precision',
                                      n_repeats=3,random_state=SEED,n_jobs=1,max_samples=min(3000,len(xv)))
    imp=pd.DataFrame({'feature':FEATURES,'mean_ap_decrease':importance.importances_mean,
                       'std_ap_decrease':importance.importances_std}).sort_values('mean_ap_decrease',ascending=False)
    imp.to_csv(output/'permutation_importance.csv',index=False)
    transformed=models[selected].named_steps['preprocess'].transform(xv.iloc[:200])
    feature_names=models[selected].named_steps['preprocess'].get_feature_names_out()
    explainer=shap.Explainer(models[selected].named_steps['model'],transformed,feature_names=feature_names)
    shap_values=explainer(transformed)
    shap_array=shap_values.values
    if shap_array.ndim == 3:
        shap_array=shap_array[:,:,1]
    mean_abs=np.abs(shap_array).mean(axis=0)
    pd.DataFrame({'feature':feature_names,'mean_abs_shap':mean_abs}).sort_values('mean_abs_shap',ascending=False).to_csv(output/'shap_feature_importance.csv',index=False)
    top_idx=np.abs(shap_array[:5]).argmax(axis=1)
    example_probs=calibrated_probability(models[selected],calibrators[selected],xv.iloc[:5])
    pd.DataFrame({'encounter_id':df.iloc[splits['validation']].encounter_id.iloc[:5].to_numpy(),
                  'probability':example_probs,
                  'prediction_at_threshold':(example_probs>=threshold).astype(int),
                  'largest_abs_shap_feature':feature_names[top_idx],
                  'largest_shap_contribution':shap_array[np.arange(5),top_idx]}).to_csv(output/'individual_prediction_examples.csv',index=False)
    joblib.dump({'pipeline':models[selected],'calibrator':calibrators[selected],
                 'features':FEATURES,'threshold':threshold},output/'model.joblib')
    (output/'run_manifest.json').write_text(json.dumps(audit,indent=2))
    make_plots(ye,probs,selected,threshold,output)
    best=next(r for r in evaluation if r['model']==selected)
    report=f'''# Readmission model evaluation

## Cohort and protocol
Source rows: {len(raw):,}. Eligible discharge encounters: {len(df):,}; excluded expired/hospice dispositions: {len(raw)-len(df):,}.
Prediction is at discharge; target is the recorded `<30` label. There are {len(FEATURES)} input fields. IDs, target, race and gender are excluded from predictors. Race and gender are retained only for an exploratory audit.
Patients are disjoint across train, calibration, validation and test sets (verified overlap: zero). No reliable calendar dates are available here, so this is not temporal or external validation. Repeated encounters within each split remain correlated.

## Held-out results
Selected model: **{selected}**, selected by validation average precision before test evaluation.
- Test encounters: {len(ye):,}; positive fraction: {ye.mean():.4f}.
- Average precision: **{best['average_precision']:.4f}**, compared with prevalence-baseline {ye.mean():.4f}.
- ROC AUC: **{best['roc_auc']:.4f}**; calibrated Brier score: **{best['brier_score']:.4f}** (lower is better).
- Validation-selected threshold: {threshold:.3f}; test precision {best['precision']:.4f}, recall {best['recall']:.4f}; flagged fraction {best['flagged_fraction']:.4f}.

Average precision is reported explicitly; it is not trapezoidal PR AUC. Calibration is fitted on a separate set. It is not guaranteed to improve test calibration; see uncalibrated and calibrated Brier values in `test_metrics.csv`. Threshold selection maximizes validation F2 and is only a demonstration, not a clinically justified operating policy.

## Interpretation and limitations
Validation permutation importance is in `permutation_importance.csv`; it measures predictive reliance, not causation. Subgroup estimates omit groups under 100 encounters or with only one label and do not establish fairness. Cluster-bootstrap intervals in `run_manifest.json` resample test patients ({args.bootstrap} repetitions); they do not include training uncertainty.
The historical diabetes-encounter sample is selected and is not representative of all patients. Readmissions outside the recorded system and incomplete follow-up may affect labels. Transfers and other non-home discharges remain, so the endpoint is a dataset benchmark rather than a clean discharge-to-home deployment cohort. No hospital IDs or dates are used for external/temporal testing. This model is not clinically validated or deployed.
'''
    (output/'evaluation.md').write_text(report,encoding='utf-8')
    print(json.dumps({'selected_model':selected,'metrics':best,'splits':audit['splits']},indent=2),flush=True)


if __name__=='__main__': main()
