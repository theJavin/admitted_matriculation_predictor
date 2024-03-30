from term import *
import requests, miceforest as mf
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, KBinsDiscretizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import f1_score
from sklearn import set_config
set_config(transform_output="pandas")

def feature_importance_df(self, dataset=0, iteration=None, normalize=True):
    targ = [self._get_var_name_from_scalar(int(i)) for i in np.sort(self.imputation_order)]
    feat = [self._get_var_name_from_scalar(int(i)) for i in np.sort(self.predictor_vars)]
    I = pd.DataFrame(self.get_feature_importance(dataset, iteration), index=targ, columns=feat).T
    return I / I.sum() * 100 if normalize else I
mf.ImputationKernel.feature_importance_df = feature_importance_df

def inspect(self, **kwargs):
    self.plot_imputed_distributions(wspace=0.3,hspace=0.3)
    plt.show()
    self.plot_mean_convergence(wspace=0.3, hspace=0.4)
    plt.show()
    I = self.feature_importance_df(**kwargs)
    I.disp(100)
    return I
mf.ImputationKernel.inspect = inspect


@dataclasses.dataclass
class AMP(MyBaseClass):
    cycle_day: int
    term_codes: typing.List
    infer_term: int
    crse: typing.List
    attr: typing.List
    styp_codes: tuple = ('n','t','t')
    fill: typing.Dict = None
    trf_grid: typing.Dict = None
    imp_grid: typing.Dict = None
    clf_grid: typing.Dict = None
    overwrite: typing.Dict = None
    show: typing.Dict = None


    def __post_init__(self):
        self.root = root_path / f"resources/rslt/{rjust(self.cycle_day,3,0)}"
        mkdir(self.root)
        D = {'trm':False, 'adm':False, 'reg':False, 'flg':False, 'raw':False, 'inputs':False,
            #   'term':False, 'raw_df':False, 'reg_df':False, 'X':False, 'Y':False,
             'transformed':False, 'imputed':False, 'predicted':False, 'performance':False, 'optimal':False, 'details':False, 'summary':False}
        
        for x in ['overwrite','show']:
            self[x] = D.copy() if self[x] is None else D.copy() | self[x]
        self.overwrite['reg_df'] = True
        self.overwrite['raw'] |= self.overwrite['reg'] | self.overwrite['adm'] | self.overwrite['flg']
        self.overwrite['inputs'] |= self.overwrite['raw']
        for k, v in self.overwrite.items():
            if v:
                delete(self.root / k)
        for k in ['fill','trf_grid','imp_grid']:
            if k not in self:
                self[k] = dict()

        self.crse = uniquify(['_allcrse', *listify(self.crse)])
        self.styp_codes = uniquify(self.styp_codes)
        self.term_codes = [x for x in uniquify(self.term_codes) if x != self.infer_term]

        self.trf_list = cartesian({k: uniquify(v, key=str) for k,v in self.trf_grid.items()})
        self.trf_list = [uniquify({k:v for k,v in t.items() if v not in ['drop',None,'']}) for t in self.trf_list]
        imp_default = {'datasets':5, 'iterations':3, 'mmc':10, 'mmf':'mean_match_default', 'tune':True}
        self.imp_list = cartesian(self.imp_grid)
        self.imp_list = [uniquify(imp_default|v) for v in self.imp_list]
        clf_default = {'datasets':5, 'iterations':3, 'mmc':10, 'mmf':'mean_match_default', 'tune':True}
        self.clf_list = cartesian(self.clf_grid)
        self.clf_list = [uniquify(clf_default | v) for v in self.clf_list]
        self.params_list = mysort([uniquify({'clf':clf, 'imp':imp, 'trf':trf}) for clf, imp, trf in it.product(self.clf_list, self.imp_list, self.trf_list)], key=str)


    def get_filename(self, path, suffix='.pkl'):
        return (self.root / join(path.values() if isinstance(path, dict) else path, '/')).with_suffix(suffix)

    def get(self, path, val=None, **kwargs):
        assert 'params' not in path
        if val is not None:
            nest(path, self.__dict__, val)
            write(self.get_filename(path, **kwargs), val, overwrite=True)
        b = False
        try:
            val = nest(path, self.__dict__)
        except:
            try:
                val = read(self.get_filename(path))
                nest(path, self.__dict__, val)
            except:
                val = dict()
                b = True
        return val, b

    def get_inputs(self):
        path = ['inputs','all']
        A, b = self.get(path)
        repl = {'term_code':'pred_code', 'term_desc':'pred_desc'}

        opts = {x:self[x] for x in ['cycle_day','overwrite','show']}
        if 'term' not in A:
            A['term'] = {term_code: TERM(term_code=term_code, **opts).get_raw() for term_code in uniquify([*self.term_codes,self.infer_term])}

        if 'raw_df' not in A:
            with warnings.catch_warnings(action='ignore'):
                A['raw_df'] = pd.concat([term.raw for term in A['term'].values()], ignore_index=True).dropna(axis=1, how='all').rename(columns=repl).prep()

        if 'reg_df' not in A:
            with warnings.catch_warnings(action='ignore'):
                A['reg_df'] = {k: pd.concat([term.reg[k].query(f"crse in {self.crse}") for term in A['term'].values()]).rename(columns=repl).prep() for k in ['cur','end']}

        where = lambda x: x.query("levl_code == 'ug' and styp_code in ('n','r','t')").copy()
        if 'X' not in A:
            R = A['raw_df'].copy()
            repl = {'ae':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'r1':1, 'r2':2, 'r3':3, 'r4':4}
            R['hs_qrtl'] = pd.cut(R['hs_pctl'], bins=[-1,25,50,75,90,101], labels=[4,3,2,1,0], right=False).combine_first(R['apdc_code'].map(repl))
            R['remote'] = R['camp_code'] != 's'
            R['resd'] = R['resd_code'] == 'r'
            R['lgcy'] = ~R['lgcy_code'].isin(['n','o'])
            R['majr_code'] = R['majr_code'].replace({'0000':pd.NA, 'und':pd.NA, 'eled':'eted', 'agri':'unda'})
            R['coll_code'] = R['coll_code'].replace({'ae':'an', 'eh':'ed', 'hs':'hl', 'st':'sm', '00':pd.NA})
            R['coll_desc'] = R['coll_desc'].replace({
                'ag & environmental_sciences':'ag & natural_resources',
                'education & human development':'education',
                'health science & human_service':'health sciences',
                'science & technology':'science & mathematics'})
            majr = ['majr_desc','dept_code','dept_desc','coll_code','coll_desc']
            S = R.sort_values('cycle_date').drop_duplicates(subset='majr_code', keep='last')[['majr_code',*majr]]
            X = where(R.drop(columns=majr).merge(S, on='majr_code', how='left')).prep().prep_bool()

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
            A['X'] = X.join(M).prep().prep_bool().set_index(self.attr, drop=False).rename(columns=lambda x:'__'+x)

        if 'Y' not in A:
            mlt_grp = ['crse','levl_code','styp_code','pred_code']
            Y = {k: A['X'][[]].join(y.set_index(['pidm','pred_code','crse'])['credit_hr']) for k, y in A['reg_df'].items()}
            agg = lambda y: where(y).groupby(mlt_grp)['credit_hr'].agg(lambda x: (x>0).sum())
            numer = agg(A['reg_df']['end'])
            denom = agg(Y['end'])
            M = (numer / denom).replace(np.inf, pd.NA).rename('mlt').reset_index().query(f"pred_code != {self.infer_term}").prep()
            M['mlt_code'] = M['pred_code']
            N = M.copy().assign(pred_code=self.infer_term)
            A['mlt'] = pd.concat([M, N], axis=0).set_index([*mlt_grp,'mlt_code'])
            Y = {k: y.squeeze().unstack().dropna(how='all', axis=1).fillna(0) for k, y in Y.items()}
            A['Y'] = Y['cur'].rename(columns=lambda x:x+'_cur').join(Y['end']>0).prep()
        
        self.get(path, A)
        for k,v in A.items():
            self[k] = v
        
        missing = [c for c in self.crse if c not in self.Y]
        assert not missing, f'missing {missing}'


    def get_model(self, X, params, inspect=False):
        params = params.copy()
        iterations = params.pop('iterations')
        datasets = params.pop('datasets')
        tune = params.pop('tune')
        mmc = params.pop('mmc')
        mmf = params.pop('mmf')
        if mmc > 0 and mmf is not None:
            params['mean_match_scheme'] = getattr(mf, mmf).copy()
            params['mean_match_scheme'].set_mean_match_candidates(mmc)
        if tune:
            # print('tuning')
            model = mf.ImputationKernel(X, datasets=1, **params)
            model.mice(iterations=1)
            optimal_parameters, losses = model.tune_parameters(dataset=0, optimization_steps=5)
        else:
            # print('not tuning')
            optimal_parameters = None
        model = mf.ImputationKernel(X, datasets=datasets, **params)
        model.mice(iterations=iterations, variable_parameters=optimal_parameters)
        if inspect:
            model.inspect()
        return model


    def get_transformed(self, path):
        path = path | {'nm':'transformed', 'crse':'_allcrse', 'train_code':'all'}
        A, b = self.get(path)
        if b:
            for params in self.params_list:
                params = subdct(params, ['trf'], True)
                p = str(params)
                if p not in A:
                    trf = ColumnTransformer([(c,t,["__"+c]) for c,t in params['trf'].items()], remainder='drop', verbose_feature_names_out=False)
                    A[p] = trf.fit_transform(self.X.query(f"styp_code == @path['styp_code']")).prep().prep_bool().prep_category().sort_index()
            self.get(path, A)
        return A, b


    def get_imputed(self, path):
        path = path | {'nm':'imputed', 'crse':'_allcrse', 'train_code':'all'}
        A, b = self.get(path)
        if b:
            for params in self.params_list:
                p = str(subdct(params, ['imp','trf'], True))
                if p not in A:
                    q = str(subdct(params, ['trf'], True))
                    T = self.get_transformed(path)[0][q]
                    imp = self.get_model(T, params['imp'])
                    A[p] = pd.concat([imp.complete_data(k).addlevel('imp', k) for k in range(imp.dataset_count())])
            self.get(path, A)
        return A, b


    def get_predicted(self, path):
        path = path | {'nm':'predicted'}
        A, b = self.get(path)
        if b:
            for params in self.params_list:
                p = str(params)
                if p not in A:
                    q = str(subdct(params, ['imp','trf'], True))
                    I = self.get_imputed(path)[0][q]
                    cols = uniquify(['_allcrse_cur', path['crse']+'_cur', path['crse']], False)
                    Z = I.join(self.Y[cols]).prep().prep_bool().prep_category().sort_index()
                    actual = Z[path['crse']].copy().rename('actual').to_frame()
                    Z.loc[Z.eval(f"pred_code!=@path['train_code']"), path['crse']] = pd.NA
                    clf = self.get_model(Z, params['clf'])
                    Z.loc[:, path['crse']] = pd.NA
                    predicted = clf.impute_new_data(Z)
                    A[p] = pd.concat([actual
                                .assign(predicted=predicted.complete_data(k)[path['crse']])
                                .addlevel('crse', path['crse'])
                                .addlevel('train_code', path['train_code'])
                                .addlevel('sim', k)
                            for k in range(predicted.dataset_count())]).prep_bool()[['predicted','actual']]
            self.get(path, A)
        return A, b


    def get_performance(self, path):
        path = path | {'nm':'performance'}
        A, b = self.get(path)
        if b:
            for params in self.params_list:
                p = str(params)
                if p not in A:
                    P = self.get_predicted(path)[0][p]
                    A[p] = 100*(P['predicted'] == P['actual']).mean()
            self.get(path, A)
        return A, b


    def get_optimal(self, path):
        path = path | {'nm':'optimal'}
        A, b = self.get(path)
        if b:
            P = self.get_performance(path)[0]
            A = min(P, key=P.get)
            self.get(path, A)
        return A, b


    def get_details(self, path):
        path = path | {'nm':'details'}
        A, b = self.get(path)
        if b:
            p = self.get_optimal(path)[0]
            A = self.get_predicted(path)[0][p]
            self.get(path, A)
        return A, b


    def get_summary(self, path):
        path = path | {'nm':'summary'}
        A, b = self.get(path)
        if b:
            D = self.get_details(path)[0].join(self.mlt).reset_index()
            T = D[['pred_code','pred_desc']].drop_duplicates()
            for k in ['train','mlt']:
                D = D.merge(T.rename(columns=lambda x: x.replace('pred',k)))
            for k in ['predicted','actual']:
                D[k+'_mlt'] = D[k] * D['mlt']
            E = D.copy().assign(styp_code='all', styp_desc='all incoming')
            A = (
                pd.concat([D,E])
                .groupby(['crse','levl_code','styp_code','styp_desc','pred_code','pred_desc','train_code','train_desc','mlt_code','mlt_desc','imp','sim'])
                .apply(lambda x: pd.Series({
                    'predicted': x['predicted_mlt'].sum(),
                    'actual': x['actual_mlt'].sum(),
                    'error': x['predicted_mlt'].sum() - x['actual_mlt'].sum(),
                    'error_pct': (x['predicted_mlt'].sum() - x['actual_mlt'].sum()) / x['actual_mlt'].sum() * 100,
                    'mse_pct': ((1*x['predicted'] - x['actual'])**2).mean()*100,
                    'f1_inv_pct': (1-f1_score(x['actual'], x['predicted'], zero_division=np.nan))*100,
                }), include_groups=False)
            )
            self.get(path, A)
        return A, b

    def run(self, nm):
        progress = [len(self.crse) * len(self.styp_codes) * len(self.term_codes), 0]
        start_time = time.perf_counter()
        w = 100
        print("=" * w)
        print(nm)
        for crse in self.crse:
            for styp_code in self.styp_codes:
                for train_code in self.term_codes:
                    path = {'nm':nm, 'crse':crse, 'styp_code':styp_code, 'train_code':train_code}
                    A, b = getattr(self, 'get_'+nm)(path)
                    progress[b] += (2*b-1)
                    elapsed = (time.perf_counter() - start_time) / 60
                    complete = progress[1] / progress[0] if progress[0] > 0 else 1
                    rate = elapsed / progress[1] if progress[1] > 0 else 0
                    remaining = rate * (progress[0] - progress[1])
                    msg = f"{join(path.values())}; complete: {progress[1]} / {progress[0]} = {complete*100:.2f}%; elapsed = {elapsed:.2f} min; remaining = {remaining:.2f} min @ {rate:.2f} min per model"
                    # if b:
                    print(msg)


    def push(self):
        target_url = 'https://prod-121.westus.logic.azure.com:443/workflows/784fef9d36024a6abf605d1376865784/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=1Yrr4tE1SwYZ88SU9_ixG-WEdN1GFicqJwH_KiCZ70M'
        with open(self.root / 'summary_df.csv', 'rb') as target_file:
            response = requests.post(target_url, files = {"amp_summary.csv": target_file})
        print('file pushed')


    def main(self):
        self.get_inputs()
        for nm in ['transformed', 'imputed', 'predicted', 'performance', 'optimal', 'details', 'summary']:
            self.run(nm)
        for path in ['details_df', 'summary_df']:
            A, b = self.get(path)
            if b:
                A = pd.concat([C for A in self[nm].values() for B in A.values() for C in B.values()])
                self.get(path, A)
                self.get(path, A, suffix='.csv')


code_desc = lambda x: [x+'_code', x+'_desc']
passthru = ['passthrough']
# passdrop = ['passthrough', 'drop']
passdrop = passthru
bintrf = lambda n_bins: KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)
pwrtrf = make_pipeline(StandardScaler(), PowerTransformer())

kwargs = {
    # 'term_codes': np.arange(2020,2025)*100+8,
    'term_codes': np.arange(2021,2024)*100+8,
    'infer_term': 202408,
    'show': {
        # 'reg':True,
        # 'adm':True,
    },
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
    'cycle_day': (TERM(term_code=202408).cycle_date-pd.Timestamp.now()).days+1,
    # 'cycle_day': 170,
    'crse': [
        # 'agec2317',
        # 'agri1100',
        # 'agri1419',
        # 'ansc1319',
        # 'arts1301',
        'biol1406',
        # # 'biol2401',
        # 'busi1301',
        # 'comm1311',
        # # 'comm1315',
        # # 'engl1301',
        # # 'govt2305',
        # # 'govt2306',
        # # 'hist1301',
        'math1314',
        # 'math1324',
        # 'math1342',
        # 'math2412',
        # 'math2413',
        # 'psyc2301',
        # 'univ0204',
        # 'univ0301',
        # 'univ0304',
        ],
    'trf_grid': {
        'act_equiv': passthru,
        'act_equiv_missing': passdrop,
        # 'admt_code': passdrop,
        'apdc_day': passthru,
        # 'appl_day': passthru,
        'birth_day': [*passthru, pwrtrf],#, ],
        # 'camp_code': passdrop,
        'coll_code': passthru,
        'distance': [*passthru, pwrtrf],#, bintrf(5)],
        # 'fafsa_app': passthru,
        # 'finaid_accepted': passthru,
        'gap_score': passthru,
        'gender': passthru,
        'hs_qrtl': passthru,
        'international': passthru,
        # 'levl_code': passthru,
        'lgcy': passthru,
        'math': passthru,
        'oriented': passthru,
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
    'imp_grid': {
        # 'mmc': 10,
        # 'datasets': 2,
        # 'iterations': 1,
        # 'tune': False,
    },
    'clf_grid': {
        # 'mmc': 10,
        # 'datasets': 2,
        # 'iterations': 1,
        # 'tune': False,
    },

    'overwrite': {
        # 'trm':True,
        # 'reg':True,
        # 'adm':True,
        # 'flg':True,
        # 'raw':True,
        # 'inputs': True,
        # 'transformed': True,
        # 'imputed': True,
        # 'predicted': True,
        # 'performance': True,
        # 'optimal': True,
        # 'details': True,
        # 'summary': True,
    },
    'styp_codes': ['n','t','r'],
}


if __name__ == "__main__":
    print(pd.Timestamp.now())

    @pd_ext
    def disp(df, max_rows=4, max_cols=200, **kwargs):
        display(HTML(df.to_html(max_rows=max_rows, max_cols=max_cols, **kwargs)))
        print(df.head(max_rows).reset_index().to_markdown(tablefmt='psql'))

    from IPython.utils.io import Tee
    kwargs['styp_codes'] = ['n','t']
    self = AMP(**kwargs)
    with contextlib.closing(Tee(self.root / 'log.txt', "w", channel="stdout")) as outputstream:
        self.main()
        # self.push()