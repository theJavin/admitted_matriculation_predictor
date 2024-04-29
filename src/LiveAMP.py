from term import *
import hashlib, miceforest as mf, flaml as fl
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, log_loss, fbeta_score, f1_score
from sklearn import set_config
set_config(transform_output="pandas")
warnings.filterwarnings("ignore", message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`")
pwrtrf = make_pipeline(StandardScaler(), PowerTransformer())

################## Metrics that work for FLAML and sklearn ##################
class Metric(MyBaseClass):
    """Parent class for metrics that work with FLAML and sklearn"""
    def __str__(self):
        return self.__repr__()
    def __lt__(self, other):
        return str(self) < str(other)
    def __call__(self, X_val, y_val, estimator, labels, X_train, y_train, weight_val=None, weight_train=None, *args):
        """for FLAML"""
        start = time.time()
        y_pred = estimator.predict(X_val)
        pred_time = (time.time() - start) / len(X_val)
        val_loss = self.score(y_val, y_pred, labels=labels, sample_weight=weight_val)
        y_pred = estimator.predict(X_train)
        train_loss = self.score(y_train, y_pred, labels=labels, sample_weight=weight_train)
        return val_loss, {
            "val_loss": val_loss,
            "train_loss": train_loss,
            "pred_time": pred_time,
        }

class F_beta(Metric):
    """Implements F_beta for FLAML https://scikit-learn.org/stable/modules/generated/sklearn.metrics.fbeta_score.html"""
    def __init__(self, beta):
        self.beta = float(beta)
    def __repr__(self):
        return f"F_{self.beta:.2f}"
    def score(self, y_true, y_pred, **kwargs):
        return 1 - fbeta_score(y_true, y_pred, beta=self.beta, **kwargs)

class Accuracy(Metric):
    def __repr__(self):
        return "accuracy"
    def score(self, y_true, y_pred, **kwargs):
        return 1 - accuracy_score(y_true, y_pred, **kwargs)

class LogLoss(Metric):
    def __repr__(self):
        return "log loss"
    def score(self, y_true, y_pred, **kwargs):
        return log_loss(y_true, y_pred, **kwargs)

################## AMP ##################
@dataclasses.dataclass
class AMP(MyBaseClass):
    cycle_day : int = (Term(term_code=202408).cycle_date-pd.Timestamp.now()).days+1
    proj_code : int = 202408
    train_code: int = 202308
    pred_codes: tuple = (202108, 202208, 202308, 202408)
    crse_code : str = '_allcrse'
    styp_code : str = 'n'
    stats: tuple = (pctl(0), pctl(25), pctl(50), pctl(75), pctl(100))
    show: set = dataclasses.field(default_factory=set)
    param: dict = dataclasses.field(default_factory=dict)    
    root_path: str = f"/home/scook/institutional_data_analytics/admitted_matriculation_projection/resources/rslt"
    dependence: dict = dataclasses.field(default_factory=lambda: {'adm':'raw', 'flg':'raw', 'raw':'X', 'reg':'X', 'X':'X_proc', 'X_proc':'Y_pred'})

    def __post_init__(self):
        super().__post_init__()
        self.root_path /= rjust(self.cycle_day,3,0)

    def get_terms(self):
        def func():
            print()
            self.terms = {
                pred_code: {
                    stage: Term(term_code=pred_code, cycle_day=cycle_day, overwrite=self.overwrite, show=self.show).get_reg().get_raw()
                for stage, cycle_day in {'cur': self.cycle_day, 'end':0}.items()}
            for pred_code in self.pred_codes}
        return self.get(func, "terms.pkl")


    def get_X(self):
        def func():
            repl = {'term_code':'pred_code', 'term_desc':'pred_desc'}
            self.reg_df = {stage: (pd.concat([
                        self.terms[pred_code][stage].reg
                        .rename(columns=repl)
                        .set_index(['pidm','levl_code','styp_code','pred_code','crse_code'])
                        ['credit_hr'].unstack()
                    for pred_code in self.pred_codes]).copy().prep()
                ) for stage in ['cur','end']}
            
            self.raw_df = pd.concat([
                        self.terms[pred_code]['cur'].raw
                        .rename(columns=repl)
                    for pred_code in self.pred_codes]).copy().dropna(axis=1, how='all').reset_index(drop=True).prep()
            R = self.raw_df.copy()
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
            X = R.drop(columns=majr).merge(S, on='majr_code', how='left').prep(bool=True)
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

            fill = {
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
                'oriented': False,
            }
            for k, v in fill.items():
                X[k] = X.impute(k, *listify(v))
            M = X.isnull().rename(columns=lambda x:x+'_missing')

            code_desc = lambda x: [x+'_code', x+'_desc']
            attr = [
                'index',
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
            ]
            self.X = X.join(M).reset_index().set_index(attr, drop=False).rename(columns=lambda x:'__'+x).prep(bool=True)
            self.Y = self.reg_df['cur'].rename(columns=lambda x:x+'_cur').join(self.reg_df['end'], how='outer')
            
            def g(A):
                cols = self.Y.columns.tolist()
                A[cols] = A[cols].astype(float).fillna(0)
                cols.remove('_allcrse_cur')
                A[cols] = A[cols] > 0
                return A

            self.Y = g(self.Y)
            self.Z = g(self.X.join(self.Y.rsindex(['pidm','pred_code']), how='left'))
            agg = lambda y: y.filter(self.reg_df['end'].columns).groupby(['styp_code','pred_code']).sum()
            N = agg(self.Y)
            D = agg(self.Z)
            M = (N / D).query("styp_code in ('n','r','t') & pred_code!=@self.proj_code")
            M[np.isnan(M) | np.isinf(M)] = pd.NA
            self.mlt = M.prep()
            del self.terms
            # del self.Y
        return self.get(func, "X.pkl", "terms")


    def get_X_proc(self):
        def func():
            trf = ColumnTransformer(self.param['trf'][2], remainder='drop', verbose_feature_names_out=False)
            self.X_trf = trf.fit_transform(self.X.query(f"styp_code==@self.styp_code")).prep(bool=True, cat=True).sample(frac=1)
            imp_dct = self.param['imp'][2].copy()
            iterations = imp_dct.pop('iterations')
            tune = imp_dct.pop('tune')
            if tune:
                ds = imp_dct.pop('datasets')
                imp = mf.ImputationKernel(self.X_trf, datasets=1, **imp_dct)
                imp.mice(iterations)
                variable_parameters, losses = imp.tune_parameters(dataset=0)
                imp_dct['datasets'] = ds
            else:
                variable_parameters = None
            imp = mf.ImputationKernel(self.X_trf, **imp_dct)
            imp.mice(iterations, variable_parameters=variable_parameters)
            self.X_proc = [
                imp.complete_data(k)
                .addlevel('trf_hash', self.param['trf'][0])
                .addlevel('imp_hash', self.param['imp'][0])
                .addlevel('sim', k)
                .prep(bool=True, cat=True)
            for k in range(imp.dataset_count())]
        return self.get(func, f"X_proc/{self.styp_code}/{self.param['trf'][0]}/{self.param['imp'][0]}.pkl", "X")


    def get_weight(self, y):
        t = y.astype(float)
        w = t.value_counts()
        w = w.sum() / w / 2
        return t.replace(w)


    def get_Y_pred(self):
        def func():
            cols = uniquify(['_allcrse_cur', self.crse_code+'_cur', self.crse_code])
            try:
                y = self.Z[cols]
                assert y.query(f"pred_code=={self.train_code}")[self.crse_code].sum() >= 10
            except:
                return None
            clf_dct = self.param['clf'][2].copy() | {'split_type':'stratified', 'task':'classification', 'verbose':0, 'log_file_name': self.path.with_suffix('.log')}
            clf_dct['log_file_name'].parent.mkdir(exist_ok=True, parents=True)
            tol = clf_dct.pop('tol')
            Z = [X.join(y).addlevel('crse_code', self.crse_code).addlevel('train_code', self.train_code).addlevel('clf_hash', self.param['clf'][0]).prep(bool=True, cat=True) for X in self.X_proc]
            Z_trn = [z.query(f"pred_code=={self.train_code}") for z in Z]
            total_true = Z_trn[0][self.crse_code].sum()

            print()
            weight_dct = clf_dct | {'time_budget':15}
            w = 0
            weight_hist = dict()
            for i in range(15):
                j = i % len(Z)
                clf = fl.AutoML(**weight_dct)
                z_trn = Z_trn[i]
                with warnings.catch_warnings(action='ignore'):
                    clf.fit(
                        z_trn.drop(columns=self.crse_code),
                        z_trn[self.crse_code],
                        sample_weight=1+(2*z_trn[self.crse_code]-1)*w,
                        **weight_dct)
                    P = [clf.predict(z.drop(columns=self.crse_code)).sum() for z in Z_trn]
                err_pct = (np.mean(P) / total_true - 1) * 100.0
                print(w, err_pct)
                weight_hist[w] = abs(err_pct)
                if abs(err_pct) < tol:
                    break
                w += (2*(err_pct<0)-1)/(2**(j+1))
            weight = min(weight_hist, key=weight_hist.get)
            print(weight, weight_hist[weight])

            self.clf_lst = []
            L = ['Z_proc', 'Y_pred', 'train_score', 'summary']
            for k in L:
                self[k+'_lst'] = []
            for z, z_trn in zip(Z, Z_trn):
                clf = fl.AutoML(**clf_dct)
                clf.weight = weight
                with warnings.catch_warnings(action='ignore'):
                    clf.fit(
                        z_trn.drop(columns=self.crse_code),
                        z_trn[self.crse_code],
                        sample_weight=1+(2*z_trn[self.crse_code]-1)*clf.weight,
                        **clf_dct)
                    clf.Y_pred = (
                        z[self.crse_code].rename('actual').to_frame()
                        .assign(predicted=clf.predict(z.drop(columns=self.crse_code)))
                        .assign(proba=clf.predict_proba(z.drop(columns=self.crse_code))[:,1])
                        .prep(bool=True)
                    ).copy()
                clf.Z_proc = z
                clf.train_score = clf.best_result['val_loss'] * 100
                clf.summary = self.summarize(clf)
                self.clf_lst.append(clf)
                for k in L:
                    self[k+'_lst'].append(getattr(clf, k))
            for k in ['Y_pred', 'summary']:
                self[k] = pd.concat(self[k+'_lst'])
            grp = [k for k in self.summary.index.names if k!= 'sim']
            self.rslt = {str(stat): self.summary.groupby(grp).agg(stat) for stat in self.stats}
        return self.get(func, f"Y_pred/{self.styp_code}/{self.crse_code}/{self.train_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl", "X_proc")


    def summarize(self, clf):
        S = clf.Y_pred.groupby(['crse_code','levl_code','styp_code','train_code','pred_code','trf_hash','imp_hash','clf_hash','sim']).apply(lambda y: pd.Series({
            'actual': y['actual'].sum(),
            'predicted': y['predicted'].sum(),
            'train_score': clf.train_score,
            'test_score': (1 - f1_score(y['actual'], y['predicted'], sample_weight=1+(2*y['actual']-1)*clf.weight))*100,
        })).prep()
        proj_mask = S.eval(f"pred_code==@self.proj_code")
        proj_col = f'{self.proj_code}_projection'
        S[proj_col] = S.loc[proj_mask, 'predicted'].squeeze()
        S = S[~proj_mask].join(self.mlt[self.crse_code].rename('mlt'))
        for k in ['predicted','actual',proj_col]:
            S[k] *= S['mlt']
        alpha = 1
        S['overall_score'] = (S['train_score'] + alpha * S['test_score']) / (1 + alpha)
        S['error'] = S['predicted'] - S['actual']
        S['error_pct'] = S['error'] / S['actual'] * 100
        return S[[proj_col,'overall_score','error_pct','error','predicted','actual','test_score','train_score']]


def get_stack(cycle_day=(Term(term_code=202408).cycle_date-pd.Timestamp.now()).days+1):
    dct = dict()
    append = lambda k, v: dct.setdefault(k,[]).append(v)
    path = AMP(cycle_day=cycle_day).root_path
    for A in (path / 'Y_pred').iterdir():
        for B in A.iterdir():
            for C in B.iterdir():
                for D in C.iterdir():
                    for E in D.iterdir():
                        for F in E.iterdir():
                            if F.suffix == '.pkl':
                                A = MyBaseClass().load(F)
                                print(F, end=": ")
                                try:
                                    for k in ['Y_pred', 'summary']:
                                        append(k, A[k])
                                    for k, v in A['rslt'].items():
                                        append(k, v)
                                    print('success')
                                except:
                                    print('FAILED')
                                # if A is None:
                                #     print('FAILED')
                                # else:
                                #     for k in ['Y_pred', 'summary']:
                                #         append(k, A[k])
                                #     for k, v in A['rslt'].items():
                                #         append(k, v)
                                #     print('success')
    dct = {k: pd.concat(v).prep() for k, v in dct.items()}

    # A = [MyBaseClass().load(F) for A in (path / 'Y_pred').iterdir() for B in A.iterdir() for C in B.iterdir() for D in C.iterdir() for E in D.iterdir() for F in E.iterdir()]
    # A = [amp for amp in A if amp is not None]
    # dct = {'amp': A}
    # dct |= {k: pd.concat([amp[k] for amp in A]).prep() for k in ['Y_pred', 'summary']}
    # dct = {k: pd.concat([amp[k] for amp in A]).prep() for k in ['Y_pred', 'summary']}
    # dct['rslt'] = {stat: pd.concat([amp['rslt'][stat] for amp in A]).prep() for stat in A[0]['rslt'].keys()}
    write(path / 'stack.pkl', dct, overwrite=True)
    return dct

param_grds = {
    'trf': {
        'act_equiv': 'passthrough',
        'act_equiv_missing': 'passthrough',
        'admt_code': 'drop',
        'apdc_day': 'passthrough',
        'appl_day': 'drop',
        'birth_day': 'passthrough',
        'camp_code': 'drop',
        'coll_code': 'passthrough',
        # 'distance': ['passthrough', pwrtrf],
        'distance': [pwrtrf],
        'fafsa_app': 'drop',
        'finaid_accepted': 'drop',
        'gap_score': 'passthrough',
        'gender': 'passthrough',
        'hs_qrtl': 'passthrough',
        'international': 'passthrough',
        'lgcy': 'passthrough',
        'math': 'passthrough',
        'oriented': 'passthrough',
        'pred_code': 'drop',
        **{f'race_{r}': 'passthrough' for r in ['american_indian','asian','black','pacific','white','hispanic']},
        'reading': 'passthrough',
        'remote': 'passthrough',
        'resd': 'passthrough',
        'schlship_app': 'passthrough',
        'ssb': 'passthrough',
        'styp_code': 'drop',
        'waiver': 'passthrough',
        'writing': 'passthrough',
    },
    'imp': {
        'random_state': 42,
        'datasets': 20,
        'iterations': 4,
        # 'datasets': 2,
        # 'iterations': 1,
        'tune': False,
        # 'tune': [False, True],
    },
    'clf': {
        'seed': 42,
        # 'metric': F_beta(1),
        'metric': 'f1',
        'early_stop': True,
        'time_budget': 30,
        # 'time_budget': 5,
        'estimator_list': [['lgbm','xgboost']],#,'catboost','histgb','extra_tree','xgb_limitdepth','rf','lrl1','lrl2','kneighbor'
        # 'ensemble': [False, True],
        'ensemble': False,
        'eval_method': 'cv',
        'n_splits': 5,
        'tol': 0.15,
    },
}

crse_codes = [
    '_allcrse',
    'agec2317',
    # 'ansc1119',
    'ansc1319',
    # 'anth2302',
    # 'anth2351',
    'arts1301',
    'arts1303',
    # 'arts1304',
    # 'arts3331',
    # 'biol1305',
    'biol1406',
    'biol1407',
    'biol2401',
    'biol2402',
    'busi1301',
    'busi1307',
    'chem1111',
    'chem1112',
    'chem1302',
    'chem1311',
    'chem1312',
    'chem1407',
    'chem1411',
    'chem1412',
    'comm1311',
    'comm1315',
    'comm2302',
    'crij1301',
    'dram1310',
    'dram2361',
    'dram4304',
    'easc2310',
    'econ1301',
    'econ2301',
    'engl1301',
    'engl1302',
    'engl2307',
    'engl2320',
    'engl2321',
    'engl2326',
    'engl2340',
    'engl2350',
    'engl2360',
    'engl2362',
    'engl2364',
    'engl2366',
    'engl2368',
    'engr2303',
    'envs1302',
    'fina1360',
    'geog1303',
    'geog1320',
    'geog1451',
    'geog2301',
    'geol1403',
    'geol1404',
    'geol1407',
    'geol1408',
    'govt2305',
    'govt2306',
    'hist1301',
    'hist1302',
    'hist2321',
    'hist2322',
    'huma1315',
    'kine2315',
    'math1314',
    'math1316',
    'math1324',
    'math1332',
    'math1342',
    'math2412',
    'math2413',
    'musi1303',
    'musi1310',
    'musi1311',
    'musi2350',
    'musi3325',
    'phil1301',
    'phil1304',
    'phil2303',
    'phil3301',
    'phys1302',
    'phys1401',
    'phys1402',
    'phys1403',
    'phys1410',
    'phys1411',
    'phys2425',
    'phys2426',
    'psyc2301',
    'soci1301',
    'soci1306',
    'soci2303',
    'univ0200',
    'univ0204',
    'univ0301',
    'univ0314',
    'univ0324',
    'univ0332',
    'univ0342',
]

formatter = lambda x: str(x).replace('\n','').replace(' ','')
hasher = lambda x, d=2: hashlib.shake_128(formatter(x).encode()).hexdigest(d)
param_dcts = dict()
for key, val in param_grds.items():
    lst = cartesian(val, sort=True, key=str)
    if key == 'trf':
        lst = [[(c,t,["__"+c]) for c,t in trf.items() if t not in ['drop', None, '']] for trf in lst]
    param_dcts[key] = [[hasher(k), formatter(k), k] for k in lst]

overwrite = [
    # 'adm',
    # 'flg',
    # 'raw',
    # 'reg',
    # 'X',
    # 'X_proc',
    # 'Y_pred',
]

def run_amp():
    for styp_code in ['n']:
        for crse_code in crse_codes:
            # for train_code in [202108, 202208, 202308]:
            for train_code in [202308]:
                for param in cartesian(param_dcts):
                    self = AMP(
                        styp_code=styp_code,
                        crse_code=crse_code,
                        train_code=train_code,
                        param=param,
                        overwrite=overwrite,
                        # cycle_day=143,
                    )
                    self.get_Y_pred()
                    # assert 1==2
    return get_stack(self.cycle_day)



if __name__ == "__main__":
    print(pd.Timestamp.now())
    # @pd_ext
    # def disp(df, max_rows=4, max_cols=200, **kwargs):
        # display(HTML(df.to_html(max_rows=max_rows, max_cols=max_cols, **kwargs)))
        # print(df.head(max_rows).reset_index().to_markdown(tablefmt='psql'))
    from IPython.utils.io import Tee
    with contextlib.closing(Tee('/home/scook/institutional_data_analytics/admitted_matriculation_projection/admitted_matriculation_predictor/log.txt', "w", channel="stdout")) as outputstream:
        run_amp()