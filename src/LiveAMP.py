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
crse_codes = [
    '_allcrse',
    'agec2317',
    'ansc1119',
    'ansc1319',
    'anth2302',
    'anth2351',
    'arts1301',
    'arts1303',
    'arts1304',
    'arts3331',
    'biol1305',
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

@dataclasses.dataclass
class AMP(MyBaseClass):
    cycle_day : int = None
    proj_code : int = 202408
    train_code: int = 202308
    pred_codes: tuple = (202108, 202208, 202308, 202408)
    crse_code : str = '_allcrse'
    styp_code : str = 'n'
    stats: tuple = (pctl(0), pctl(25), pctl(50), pctl(75), pctl(100), )
    show: set = dataclasses.field(default_factory=set)
    param: dict = dataclasses.field(default_factory=dict)    
    root_path: str = f"/home/scook/institutional_data_analytics/admitted_matriculation_projection/resources/rslt"
    dependence: dict = dataclasses.field(default_factory=lambda: {'adm':'raw', 'flg':'raw', 'raw':'X', 'reg':'X', 'X':'X_proc', 'X_proc':'Y'})

    def __post_init__(self):
        super().__post_init__()
        self.cycle_day = (Term(term_code=202408).cycle_date-pd.Timestamp.now()).days+1 if self.cycle_day is None else self.cycle_day
        self.root_path /= rjust(self.cycle_day,3,0)

    def get_X(self):
        def func():
            print()
            terms = {cycle_day: {pred_code:
                    Term(term_code=pred_code, cycle_day=cycle_day, overwrite=self.overwrite, show=self.show).get_reg().get_raw()
                for pred_code in self.pred_codes} for cycle_day in [0, self.cycle_day]}
            ren = {'term_code':'pred_code', 'term_desc':'pred_desc', 'credit_hr':'enrolled'}
            R = pd.concat([T.raw.rename(columns=ren) for pred_code, T in terms[self.cycle_day].items()]).copy().dropna(axis=1, how='all').reset_index(drop=True).prep()
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

            y_end = pd.concat([
                    T.reg
                    .rename(columns=ren)
                    .assign(crse_code=lambda X: X['crse_code'] + ('_cur' if cycle_day > 0 else ''))
                    .set_index(['pidm','levl_code','styp_code','pred_code','crse_code'])
                    [['enrolled']]
                    .fillna(0)
                for cycle_day, dct in terms.items() for pred_code, T in dct.items()]).copy().fillna(0).prep()
            y_end.loc[y_end.eval("crse_code!='_allcrse_cur' & enrolled>0")] = 1
            self.y = y_end.rsindex(['pidm','pred_code','crse_code'])
            y_cur = self.X[[]].join(self.y)
            agg = lambda y: y.groupby(['styp_code','pred_code','crse_code']).sum().query(f"styp_code in ('n','r','t') and pred_code!={self.proj_code} and not crse_code.str.contains('cur')")
            # N = agg(y_end)
            # D = agg(y_cur)
            self.mlt = agg(y_end).join(agg(y_cur), how='outer', lsuffix='_end', rsuffix='_cur').fillna(0)
            self.mlt['mlt'] = np.where(self.mlt.min(axis=1)>0, self.mlt['enrolled_end'] / self.mlt['enrolled_cur'], pd.NA)
            
            # self.mlt_num = agg(y_end)
            # self.mlt_den = agg(y_cur)
            # # self.mlt = (self.mlt_num / self.mlt_den).dropna().squeeze().rename('mlt')
            # self.mlt = (agg(y_end) / agg(y_cur)).dropna().squeeze().rename('mlt')
        return self.get(func, "X.pkl")


    def get_X_proc(self):
        def func():
            trf = ColumnTransformer(self.param['trf'][2], remainder='drop', verbose_feature_names_out=False)
            X_trf = trf.fit_transform(self.X.query("styp_code==@self.styp_code")).prep(bool=True, cat=True).sample(frac=1)
            imp_dct = self.param['imp'][2].copy()
            iterations = imp_dct.pop('iterations')
            tune = imp_dct.pop('tune')
            if tune:
                ds = imp_dct.pop('datasets')
                imp = mf.ImputationKernel(X_trf, datasets=1, **imp_dct)
                imp.mice(iterations)
                variable_parameters, losses = imp.tune_parameters(dataset=0)
                imp_dct['datasets'] = ds
            else:
                variable_parameters = None
            imp = mf.ImputationKernel(X_trf, **imp_dct)
            imp.mice(iterations, variable_parameters=variable_parameters)
            self.X_proc = pd.concat([
                    imp.complete_data(k)
                    .addlevel({'trf_hash':self.param['trf'][0], 'imp_hash':self.param['imp'][0], 'sim':k})
                    .prep(bool=True, cat=True)
                for k in range(imp.dataset_count())])
            del self.X
        return self.get(func, f"X_proc/{self.styp_code}/{self.param['trf'][0]}/{self.param['imp'][0]}.pkl", "X")


    def get_Y(self):
        def func():
            clf_dct = self.param['clf'][2] | {'task':'classification', 'verbose':0}#, 'log_type': 'all'}
            min_calibrate = clf_dct.pop('min_calibrate')
            time_calibrate = clf_dct.pop('time_calibrate')
            max_calibrate = time_calibrate // clf_dct['time_budget']

            cols = uniquify(['_allcrse_cur', self.crse_code+'_cur', self.crse_code])
            y = self.y.query(f"crse_code.isin({cols})").unstack().droplevel(0,1)
            Z = self.X_proc.join(y).fillna({c:0 for c in cols}).addlevel({'crse_code':self.crse_code, 'train_code':self.train_code, 'clf_hash':self.param['clf'][0]}).prep(bool=True, cat=True)

            if self.crse_code not in Z or Z.query(f"pred_code=={self.train_code} & sim==0")[self.crse_code].sum() < 10:
                self.Y = Z[self.crse_code].rename('actual').to_frame().assign(proba=0.0).copy()
                self.weight = 0
                self.train_score = np.inf
            else:
                X = Z.query(f"pred_code=={self.train_code}").copy()
                t = X.query(f"sim==0").groupby([self.crse_code,'__coll_code'], observed=True)
                y = X.pop(self.crse_code)
                del self.X_proc
                del self.y
                
                def train(wgt, iter=''):
                    mask = X.reset_index()['index'].isin(t.sample(frac=0.75, random_state=clf_dct['seed']).reset_index()['index']).values
                    dct = clf_dct | {
                        'X_train':X[mask],
                        'y_train':y[mask],
                        'X_val':X[~mask],
                        'y_val':y[~mask],
                        'sample_weight':1+(2*y[mask]-1)*wgt,
                        'sample_weight_val':1+(2*y[~mask]-1)*wgt,
                        'log_file_name': self.path.with_stem(f"{self.path.stem}{iter}").with_suffix('.log'),
                    }
                    mkdir(dct['log_file_name'].parent)
                    clf = fl.AutoML(**dct)
                    clf.fit(**dct)
                    X_all = Z.copy()
                    clf.Y = X_all.pop(self.crse_code).rename('actual').to_frame().assign(proba=clf.predict_proba(X_all)[:,1]).prep(bool=True).copy()
                    return clf

                wgt = 0
                err = 0
                best_wgt = wgt
                best_rmse = np.inf
                self.hist = dict()
                print()
                for i in range(max_calibrate):
                    wgt = np.clip(wgt - err*(0.95**i), -1, 1)
                    clf = train(wgt)
                    S = clf.Y.groupby('pred_code').sum().query(f"pred_code!={self.proj_code}")
                    S['proba'] *= S['actual'] > 0
                    S['err'] = S['proba'] - S['actual']
                    err = S['err'].sum() / S['actual'].sum()

                    self.hist[wgt] = err
                    W = np.array(list(self.hist.keys  ())[-min_calibrate:])
                    E = np.array(list(self.hist.values())[-min_calibrate:])
                    rmse = np.sqrt(np.mean(E**2))
                    if min_calibrate <= len(E) and rmse < best_rmse:
                        best_wgt = W.mean()
                        best_rmse = rmse
                    print(rjust(i,3), f'wgt={wgt: .12f}', f'best_wgt={best_wgt: .12f}', f'err={err: .12f}', f'rmse={rmse:.12f}', f'best_rmse={best_rmse:.12f}')
                    if best_rmse < 0.00001:
                        break
                self.weight = best_wgt
                clf_dct['time_budget'] *= 20
                self.clf = train(best_wgt)
                self.Y = clf.Y
                self.train_score = clf.best_result['val_loss'] * 100
                self.clf = clf._trained_estimator
            self.summarize()
            # del self.mlt
        return self.get(func, f"Y/{self.styp_code}/{self.crse_code}/{self.train_code}/{self.param['trf'][0]}/{self.param['imp'][0]}/{self.param['clf'][0]}.pkl", "X_proc")


    def summarize(self):
        S = self.Y.groupby(['crse_code','levl_code','styp_code','train_code','pred_code','trf_hash','imp_hash','clf_hash','sim']).apply(lambda y: pd.Series({
            'actual': y['actual'].sum(),
            # 'predicted': y['predicted'].sum(),
            'predicted': y['proba'].sum(),
            'train_score': self.train_score,
            'test_score': log_loss(y['actual'], y['proba'], labels=[False,True], sample_weight=1+(2*y['actual']-1)*self.weight) * 100,
            'weight': self.weight,
        })).prep()
        proj_mask = S.eval(f"pred_code==@self.proj_code")
        proj_col = f'{self.proj_code}_projection'
        S = (
            S[~proj_mask]
            .join(S[proj_mask]['predicted'].droplevel('pred_code').rename(proj_col))
            .join(self.mlt)
            .sort_index()
        )
        for k in ['predicted','actual',proj_col]:
            S[k] *= S['mlt']
        S[proj_col+'_pct_change'] = (S[proj_col] / S.groupby(S.index.names.difference({'pred_code'})).transform('last')['actual'] - 1) * 100
        alpha = 1
        S['overall_score'] = (S['train_score'] + alpha * S['test_score']) / (1 + alpha)
        S['error'] = S['predicted'] - S['actual']
        S['error_pct'] = S['error'] / S['actual'] * 100
        # self.summary = S.query('actual>=10')[[proj_col,proj_col+'_pct_change','predicted','actual','error','error_pct','mlt','overall_score','test_score','train_score','weight']].dropna()
        self.summary = S[[proj_col,proj_col+'_pct_change','predicted','actual','error','error_pct','mlt','overall_score','test_score','train_score','weight']].dropna()
        grp = [k for k in self.summary.index.names if k!= 'sim']
        self.rslt = {str(stat): self.summary.groupby(grp).agg(stat) for stat in self.stats}
        self.rslt[' 50%'].disp(100)

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
        'datasets': 10,
        'iterations': 4,
        # 'datasets': 2,
        # 'iterations': 1,
        'tune': False,
        # 'tune': [False, True],
    },
    'clf': {
        'seed': 42,
        'metric': 'log_loss',
        'early_stop': True,
        'time_budget': 10,
        'estimator_list': [['lgbm','xgboost']],#,'catboost','histgb','extra_tree','xgb_limitdepth','rf','lrl1','lrl2','kneighbor'
        # 'ensemble': [False, True],
        'ensemble': False,
        'min_calibrate': 10,
        'time_calibrate': 8*60,
    },
}


formatter = lambda x: str(x).replace('\n','').replace(' ','')
hasher = lambda x, d=2: hashlib.shake_128(formatter(x).encode()).hexdigest(d)
param_dcts = dict()
for key, val in param_grds.items():
    lst = cartesian(val, sort=True, key=str)
    if key == 'trf':
        lst = [[(c,t,["__"+c]) for c,t in trf.items() if t not in ['drop', None, '']] for trf in lst]
    param_dcts[key] = [[hasher(k), formatter(k), k] for k in lst]


def run_amp(cycle_days=None, styp_codes=['n'], train_codes=[202108,202208,202308], overwrite=None):
    for kwargs in cartesian({
        'cycle_day': cycle_days,
        'styp_code': styp_codes,
        'crse_code': crse_codes,
        'train_code': train_codes,
        'param': cartesian(param_dcts),
        'overwrite': [listify(overwrite)],
        }):
            self = AMP(**kwargs)
            # self.get_X()
            # self.mlt.disp()
            # assert 1==2
            # self.get_X_proc()
            
            self.get_Y()
    return self


def get_stack(cycle_day, ext=None):
    self = AMP(cycle_day=cycle_day)
    def func():
        self.stack = dict()
        append = lambda k, v: self.stack.setdefault(k,[]).append(v)
        for fn in (self.root_path / 'Y').rglob('*.pkl'):
            A = AMP().load(fn, force=True)
            # A.mlt = self.mlt.copy()
            # A.summarize()
            # A.dump(fn)
            for k in ['Y', 'summary']:
                append(k, A[k])
            for k, v in A['rslt'].items():
                append(k, v)
        self.stack = {k: pd.concat(v).prep() for k, v in self.stack.items()}
        self.report = (
            self.stack[' 50%']
            .query("trf_hash=='fa15'")
            .reset_index()
            .drop(columns=['trf_hash','imp_hash','clf_hash'])
            .sort_values(['crse_code','levl_code','styp_code','train_code','pred_code'], ascending=[True,True,True,False,False])
            .round(2)
            .prep()
        )
        self.report.to_csv(self.root_path / f'AMP{ext}.csv', index=False)
    return self.get(func, f"stack.pkl", "X")


if __name__ == "__main__":
    print(pd.Timestamp.now())
    delattr(pd.Series, 'disp')
    delattr(pd.DataFrame, 'disp')
    @pd_ext
    def disp(df, max_rows=4, max_cols=200, **kwargs):
        print(df.reset_index().drop(columns='index', errors='ignore').head(max_rows).to_markdown(tablefmt='psql'))

    from IPython.utils.io import Tee
    with contextlib.closing(Tee('/home/scook/institutional_data_analytics/admitted_matriculation_projection/admitted_matriculation_predictor/log.txt', "w", channel="stdout")) as outputstream:
        run_amp(131)