from term import *
import requests, hashlib, miceforest as mf, flaml as fl
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, KBinsDiscretizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, auc
from sklearn.linear_model import LogisticRegression
from sklearn import set_config
set_config(transform_output="pandas")

@dataclasses.dataclass
class AMP(MyBaseClass):
    cycle_day: int = 0
    attr: tuple = ('pidm', 'styp_code')
    crse_codes: tuple = ('_allcrse',)
    styp_codes: tuple = ('n', 't', 't')
    pred_codes: tuple = (202108, 202208, 202308, 202408)
    train_codes: tuple = (202108, 202208, 202308)
    proj_code: int = 202408
    fill: dict = dataclasses.field(default_factory=dict)
    overwrite: dict = dataclasses.field(default_factory=dict)
    show: dict = dataclasses.field(default_factory=dict)
    trf_grid: dict = dataclasses.field(default_factory=dict)
    imp_grid: dict = dataclasses.field(default_factory=dict)
    clf_grid: dict = dataclasses.field(default_factory=dict)
    impute_yearly: bool = False
    n_splits: int = 3
    random_state: int = 42

    def __post_init__(self):
        self.root_path = root_path / f"resources/rslt/{rjust(self.cycle_day,3,0)}/{self.impute_yearly}"
        self.dependence = {
            'adm':'raw', 'flg':'raw', 'dst':'raw', 'raw':'terms', 'reg':'terms', 'terms':{'raw_df','reg_df'}, 'raw_df':'X', 'reg_df':{'Y','mlt'}, 
            'X':{'Y','mlt','X_proc'}, 'Y':'Z_proc', 'X_proc':'Z_proc', 'Z_proc':'Y_pred', 'Y_pred':'summary', 'summary':'optimal'}
        super().__post_init__()
        for nm in ['terms','raw_df','reg_df','X','Y','mlt','X_proc','Z_proc','Y_pred','summary','optimal']:
            self.path[nm] = self.root_path / nm
        for k in self.overwrite:
            delete(self.root_path / k)
        self.mlt_grp = ['crse_code','levl_code','styp_code','pred_code','hack_code','train_code','mlt_code','trf_hash','clf_hash','imp']
        self.crse_codes = ['_allcrse', *listify(self.crse_codes)]
        self.proj_code = listify(self.proj_code)[0]
        self.pred_codes = [*listify(self.pred_codes), *listify(self.train_codes), self.proj_code]
        for k in ['crse_codes','styp_codes','pred_codes','train_codes','attr']:
            self[k] = uniquify(self[k])
        if self.proj_code in self.train_codes:
            self.train_codes.remove(self.proj_code)
        for k in ['trf','imp','clf']:
            self[k+'_list'] = cartesian({k: uniquify(v, key=str) for k,v in self[k+'_grid'].items()})
            if k == 'trf':
                self[k+'_list'] = [ColumnTransformer([(c,t,["__"+c]) for c,t in trf.items() if t not in ['drop',None,'']], remainder='drop', verbose_feature_names_out=False) for trf in self[k+'_list']]
            self[k+'_dict'] = {hashlib.shake_128(str(trf).encode()).hexdigest(2): trf for trf in self[k+'_list']}

    def where(self, df):
        return df.rename(columns={'term_code':'pred_code', 'term_desc':'pred_desc'}).query(f"levl_code=='ug' & styp_code in ('n','r','t')").copy()

    def get_term(self, path, **kwargs):
        print()
        path.pop('nm')
        kwargs = {x:self[x] for x in ['overwrite','show']}
        return TERM(**path, **deepcopy(kwargs)).run()
    
    def get_terms(self, path, **kwargs):
        print()
        return {pred_code: {stage: self.get({'nm':'term', 'term_code': pred_code, 'cycle_day': cycle_day}) for stage, cycle_day in {'cur': self.cycle_day, 'end':0}.items()} for pred_code in self.pred_codes}
    
    def get_raw_df(self, path, **kwargs):
        return pd.concat([self.where(self.get({'nm':'terms', 'pred_code':pred_code, 'stage':'cur'}).get('raw')) for pred_code in self.pred_codes], ignore_index=True).dropna(axis=1, how='all').prep()

    def get_reg_df(self, path, **kwargs):
        return {stage: pd.concat([self.where(self.get({'nm':'terms', 'pred_code':pred_code, 'stage':stage}).get('reg').query(f"crse_code in {self.crse_codes}")) for pred_code in self.pred_codes], ignore_index=True).prep().set_index(['pidm','pred_code','crse_code']) for stage in ['cur','end']}

    def get_X(self, path, **kwargs):
        R = self.get('raw_df').copy()
        repl = {'ae':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'r1':1, 'r2':2, 'r3':3, 'r4':4}
        R['hs_qrtl'] = pd.cut(R['hs_pctl'], bins=[-1,25,50,75,90,101], labels=[4,3,2,1,0], right=False).combine_first(R['apdc_code'].map(repl))
        R['remote'] = R['camp_code'] != 's'
        R['resd'] = R['resd_code'] == 'r'
        R['oriented'] = R['oriented'] != 'n'
        R['lgcy'] = ~R['lgcy_code'].isin(['n','o'])
        R['majr_code'] = R['majr_code'].replace({'0000':pd.NA, 'und':pd.NA, 'eled':'eted', 'agri':'unda'})
        R['coll_code'] = R['coll_code'].replace({'ae':'an', 'eh':'ed', 'hs':'hl', 'st':'sm', '00':pd.NA})
        R['coll_desc'] = R['coll_code'].map({
            'an': 'ag & natural_resources',
            'ba': 'business',
            'ed': 'education',
            'en': 'engineering',
            'hl': 'health sciences',
            'la': 'liberal & fine arts',
            'sm': 'science & mathematics',
            pd.NA: 'no college designated',
        })
        majr = ['majr_desc','dept_code','dept_desc','coll_code','coll_desc']
        S = R.sort_values('cycle_date').drop_duplicates(subset='majr_code', keep='last')[['majr_code',*majr]]
        X = R.drop(columns=majr).merge(S, on='majr_code', how='left').prep().prep_bool()
        checks = [
            'cycle_day >= 0',
            'apdc_day >= cycle_day',
            'appl_day >= apdc_day',
            'birth_day >= appl_day',
            'birth_day >= 5000',
            'distance >= 0',
            'hs_pctl >=0',
            'hs_pctl <= 100',
            'hs_qrtl >= 0',
            'hs_qrtl <= 4',
            'act_equiv >= 1',
            'act_equiv <= 36',
            'gap_score >= 0',
            'gap_score <= 100',
        ]
        for check in checks:
            mask = X.eval(check)
            assert mask.all(), [check,X[~mask].disp(5)]
        for k, v in self.fill.items():
            X[k] = X.impute(k, *listify(v))
        M = X.isnull().rename(columns=lambda x:x+'_missing')
        X = X.join(M).prep().prep_bool().set_index(self.attr, drop=False).rename(columns=lambda x:'__'+x)
        return X

    def get_Y(self, path, **kwargs):
        Y = {k: self.get('X')[[]].join(y)['credit_hr'].unstack().dropna(how='all', axis=1).fillna(0) for k, y in self.get('reg_df').items()}
        Y = {k: y.assign(**{c:0 for c in self.crse_codes if c not in y.columns})[self.crse_codes] for k, y in Y.items()}
        return Y['cur'].rename(columns=lambda x:x+'_cur').join(Y['end']>0).prep()

    def get_mlt(self, path, **kwargs):
        agg = lambda y: y.query(f"pred_code!=@self.proj_code").grpby(self.mlt_grp)['credit_hr'].agg(lambda x: (x>0).sum()).rename('mlt')
        N = self.get('reg_df')['end']
        D = self.get('X')[[]].join(N)[['credit_hr']]
        N = agg(N)
        D = agg(D)
        return (N / D).replace(np.inf, pd.NA).prep()
        # A = (N / D).replace(np.inf, pd.NA).reset_index().prep()
        # B = A.rename(columns={'credit_hr':'mlt_actual'}).assign(hack_code=lambda X:X['pred_code'])
        # C = A.rename(columns={'credit_hr':'mlt_predicted', 'pred_code':'mlt_code'})
        # M = B.sindex(self.mlt_grp).join(C.sindex(self.mlt_grp)).reset_index()
        # return pd.concat([M, M.assign(pred_code=self.proj_code)]).sindex(self.mlt_grp)

    def get_X_proc(self, path, **kwargs):
        X = self.get('X')
        mask = X.eval(f"styp_code==@path['styp_code']")
        trf = self.trf_dict[path['trf_hash']]
        X_trf = trf.fit_transform(X[mask]).sample(frac=1).prep().prep_bool().prep_category()
        imp_par = self.imp_dict[path['imp_hash']].copy()
        imp_par['random_state'] = self.random_state
        iterations = imp_par.pop('iterations')
        tune = imp_par.pop('tune')
        if tune:
            ds = imp_par.pop('datasets')
            imp = mf.ImputationKernel(X_trf, datasets=1, **imp_par)
            imp.mice(1)
            imp_par['variable_parameters'], losses = imp.tune_parameters(dataset=0)
            imp_par['datasets'] = ds
        imp = mf.ImputationKernel(X_trf, **imp_par)
        imp.mice(iterations)
        X_proc = pd.concat([imp.complete_data(k).addlevel('trf_hash', path['trf_hash']).addlevel('imp', k) for k in range(imp.dataset_count())])
        return {'path': path, 'X_proc': X_proc, 'trf_str': str(trf).replace('\n','').replace(' ','')}

    def get_Z_proc(self, path, **kwargs):
        dct = self.get(path | {'nm':'X_proc'})
        dct['Z_proc'] = dct['X_proc'].join(self.get('Y')).sample(frac=1).prep().prep_bool().prep_category()
        return dct

    def get_Y_pred(self, path, **kwargs):
        dct = self.get(path | {'nm':'Z_proc', 'crse_code':'all', 'train_code':'all', 'clf_hash':'all'})
        targ = path['crse_code']
        df = dct['Z_proc']
        if df.query(f"pred_code==@path['train_code'] & imp==0")[targ].sum() < 10:
            return dct
        clf_par = self.clf_dict[path['clf_hash']]
        clf_par['seed'] = self.random_state
        cols = uniquify([*df.filter(like='__').columns, '_allcrse_cur', targ+'_cur', targ])
        df = df.filter(cols).copy().addlevel('crse_code', targ).addlevel('train_code', path['train_code']).addlevel('clf_hash', path['clf_hash']).prep().prep_category()
        for imp, Z in df.groupby('imp'):
            trn_mask = Z.eval(f"pred_code==@path['train_code']")
            clf = fl.AutoML(**clf_par)
            with warnings.catch_warnings(action='ignore'):
                clf.fit(Z[trn_mask].drop(columns=targ), Z[trn_mask][targ], eval_method='cv', n_splits=self.n_splits, **clf_par)
            Y = (
                Z[targ].rename('actual').to_frame()
                .assign(predicted=clf.predict(Z.drop(columns=targ)))
                .assign(proba=clf.predict_proba(Z.drop(columns=targ))[:,1])
            )
            append(dct, 'Y_pred', Y)
            append(dct, 'train_score', clf.best_result['val_loss'] * 100)
        return dct

    def get_summary(self, path, **kwargs):
        dct = self.get(path | {'nm':'Y_pred'})
        if 'Y_pred' not in dct:
            return dct
        for Y in dct['Y_pred']:
            trn_mask = Y.eval(f"pred_code==@path['train_code']")
            proj_mask = Y.eval(f"pred_code==@self.proj_code")
            holdout_mask = ~(trn_mask | proj_mask)
            proj = Y[proj_mask]['predicted'].sum()
            M = self.get('mlt').query(f"styp_code==@path['styp_code'] & crse_code==@path['crse_code']").rsindex('pred_code').squeeze()
            append(dct, 'holdout_score', (1 - f1_score(Y[holdout_mask]['actual'], Y[holdout_mask]['predicted'])) * 100)
            append(dct, '2024_projection_raw', proj)
            for k, v in M.items():
                append(dct, f'2024_projection_{k}', proj * v)
        dct['rslt'] = pd.DataFrame({k:v for k,v in dct.items() if 'score' in k or 'projection' in k})
        dct['summary'] = dct['rslt'].describe().T
        dct['score'] = {'train': dct['rslt']['train_score'].median(), 'holdout': dct['rslt']['holdout_score'].median()}
        dct['score']['agg'] = sum(dct['score'].values())
        return dct
    
    def get_optimal(self, path, **kwargs):
        dct = self.get(path | {'nm': 'summary'})
        A = [C for trf_hash, T in dct.items() for imp_hash, I in T.items() for clf_hash, C in I.items()]
        a = min(A, key=lambda x: x['score']['agg'])
        return a


    def run(self):
        self.get('terms')
        self.get('raw_df')
        self.get('reg_df')
        self.get('X')
        self.get('Y')
        self.get('mlt')
        for path in cartesian({'nm': '', 'styp_code': self.styp_codes, 'crse_code': self.crse_codes, 'train_code': self.train_codes, 'trf_hash': self.trf_dict.keys(), 'imp_hash': self.imp_dict.keys(), 'clf_hash': self.clf_dict.keys()}, sort=False):
            qath = path | {'crse_code':'all', 'train_code':'all', 'clf_hash':'all'}
            dct = self.get(qath | {'nm':'X_proc'})
            dct = self.get(qath | {'nm':'Z_proc'})
            dct = self.get(path | {'nm':'Y_pred'})
            dct = self.get(path | {'nm':'summary'})
            # dct['summary'].disp(100)
            # print(dct['score'])
        for path in cartesian({'nm': '', 'styp_code': self.styp_codes, 'crse_code': self.crse_codes, 'train_code': self.train_codes}, sort=False):
            dct = self.get(path | {'nm':'optimal'})
            dct['summary'].disp(100)

code_desc = lambda x: [x+'_code', x+'_desc']
pwrtrf = make_pipeline(StandardScaler(), PowerTransformer())
passthru = ['passthrough']
passdrop = ['passthrough', 'drop']
passpwr = ['passthrough', pwrtrf]

# passdrop = ['passthrough']
# passpwr = [pwrtrf]

kwargs = {
    'fill': {
        'birth_day': ['median',['pred_code','styp_code']],
        'remote': False,
        'international': False,
        **{f'race_{r}': False for r in ['american_indian','asian','black','pacific','white','hispanic']},
        'lgcy': False,
        'resd': False,
        'waiver': False,
        'fafsa_app': False,
        'schlship_app': False,
        'finaid_accepted': False,
        'ssb': False,
        'math': False,
        'reading': False,
        'writing': False,
        'gap_score': 0,
        'oriented': 'n',
    },
    'attr': [
        'pidm',
        *code_desc('pred'),
        *code_desc('apdc'),
        *code_desc('levl'),
        *code_desc('styp'),
        *code_desc('admt'),
        *code_desc('camp'),
        *code_desc('coll'),
        *code_desc('dept'),
        *code_desc('majr'),
        *code_desc('cnty'),
        *code_desc('stat'),
        *code_desc('natn'),
        *code_desc('resd'),
        *code_desc('lgcy'),
        'international',
        'gender',
        *[f'race_{r}' for r in ['american_indian','asian','black','pacific','white','hispanic']],
        'waiver',
        'birth_day',
        'distance',
        'hs_qrtl',
    ],
    'trf_grid': {
        'act_equiv': passthru,
        # 'act_equiv_missing': passdrop,
        'act_equiv_missing': passthru,
        # 'admt_code': passdrop,
        'apdc_day': passthru,
        # 'appl_day': passthru,
        # 'birth_day': passpwr,
        'birth_day': passpwr,
        # 'camp_code': passdrop,
        'coll_code': passdrop,
        'distance': passpwr,
        # 'fafsa_app': passthru,
        # 'finaid_accepted': passthru,
        'gap_score': passthru,
        'gender': passthru,
        'hs_qrtl': passthru,
        'international': passthru,
        'lgcy': passthru,
        'math': passthru,
        'oriented': passthru,
        # 'pred_code': passthru,
        **{f'race_{r}': passthru for r in ['american_indian','asian','black','pacific','white','hispanic']},
        'reading': passthru,
        'remote': passthru,
        'resd': passthru,
        'schlship_app': passthru,
        'ssb': passthru,
        # 'styp_code': passthru,
        'waiver': passthru,
        'writing': passthru,
        },
    'crse_codes': [
        # 'agec2317',
        # 'ansc1119',
        # 'ansc1319',
        # 'anth2302',
        # 'anth2351',
        # 'arts1301',
        # 'arts1303',
        # 'arts1304',
        # 'arts3331',
        # 'biol1305',
        # 'biol1406',
        # 'biol1407',
        # 'biol2401',
        # 'biol2402',
        # 'busi1301',
        # 'busi1307',
        # 'chem1111',
        # 'chem1112',
        # 'chem1302',
        # 'chem1311',
        # 'chem1312',
        # 'chem1407',
        # 'chem1411',
        # 'chem1412',
        # 'comm1311',
        # 'comm1315',
        # 'comm2302',
        # 'crij1301',
        # 'dram1310',
        # 'dram2361',
        # 'dram4304',
        # 'easc2310',
        # 'econ1301',
        # 'econ2301',
        # 'engl1301',
        # 'engl1302',
        # 'engl2307',
        # 'engl2320',
        # 'engl2321',
        # 'engl2326',
        # 'engl2340',
        # 'engl2350',
        # 'engl2360',
        # 'engl2362',
        # 'engl2364',
        # 'engl2366',
        # 'engl2368',
        # 'engr2303',
        # 'envs1302',
        # 'fina1360',
        # 'geog1303',
        # 'geog1320',
        # 'geog1451',
        # 'geog2301',
        # 'geol1403',
        # 'geol1404',
        # 'geol1407',
        # 'geol1408',
        # 'govt2305',
        # 'govt2306',
        # 'hist1301',
        # 'hist1302',
        # 'hist2321',
        # 'hist2322',
        # 'huma1315',
        # 'kine2315',
        # 'math1314',
        # 'math1316',
        # 'math1324',
        # 'math1332',
        # 'math1342',
        # 'math2412',
        # 'math2413',
        # 'musi1303',
        # 'musi1310',
        # 'musi1311',
        # 'musi2350',
        # 'musi3325',
        # 'phil1301',
        # 'phil1304',
        # 'phil2303',
        # 'phil3301',
        # 'phys1302',
        # 'phys1401',
        # 'phys1402',
        # 'phys1403',
        # 'phys1410',
        # 'phys1411',
        # 'phys2425',
        # 'phys2426',
        # 'psyc2301',
        # 'soci1301',
        # 'soci1306',
        # 'soci2303',
        # 'univ0200',
        # 'univ0204',
        # 'univ0301',
        # 'univ0314',
        # 'univ0324',
        # 'univ0332',
        # 'univ0342',
        ],
    'overwrite': {
        'trm',
        'adm',
        'flg',
        'raw',
        'reg',
        'terms',
        'raw_df',
        'reg_df',
        'X',
        'Y',
        'mlt',
        'X_proc',
        'Z_proc',
        'Y_pred',
        'summary',
        'optimal',
    },
    'clf_grid': {
        'task': 'classification',
        'verbose': 0,
        'metric': 'f1',
        'split_type': 'stratified',
        # 'early_stop': 10,
        'time_budget': 5,
        # "ensemble": True,
        # "ensemble": {
        #     "final_estimator": LogisticRegression(),
            # "passthrough": False,
        # },
    },
    'imp_grid': {
        'datasets': 30,
        'iterations': 10,
        # 'datasets': 2,
        # 'iterations': 3,
        'tune': False,
    },
    'cycle_day': (TERM(term_code=202408).cycle_date-pd.Timestamp.now()).days+1,
    # 'cycle_day': 152,
    'styp_codes':'n',
    'n_splits': 3,
    'random_state': 42,
    'train_codes': 202308,
}

if __name__ == "__main__":
    print(pd.Timestamp.now())

    @pd_ext
    def disp(df, max_rows=4, max_cols=200, **kwargs):
        display(HTML(df.to_html(max_rows=max_rows, max_cols=max_cols, **kwargs)))
        print(df.head(max_rows).reset_index().to_markdown(tablefmt='psql'))

    from IPython.utils.io import Tee
    with contextlib.closing(Tee('/home/scook/institutional_data_analytics/admitted_matriculation_projection/admitted_matriculation_predictor/log.txt', "w", channel="stdout")) as outputstream:
        self = AMP(**kwargs)
        self.run()