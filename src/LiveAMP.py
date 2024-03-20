from term import *
import miceforest as mf
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
    fill: typing.Dict = None
    trf_grid: typing.Dict = None
    imp_grid: typing.Dict = None
    overwrite: typing.Dict = None
    show: typing.Dict = None
    inspect: bool = False

    def dump(self):
        return write(self.rslt, self, overwrite=True)

    def __post_init__(self):
        self.rslt = root_path / f"resources/rslt/{rjust(self.cycle_day,3,0)}/rslt.pkl"
        mkdir(self.rslt.parent)
        D = {'trm':False, 'adm':False, 'reg':False, 'flg':False, 'raw':False, 'term':False, 'raw_df':False, 'reg_df':False, 'X':False, 'Y':False, 'pred':False}
        for x in ['overwrite','show']:
            self[x] = D.copy() if self[x] is None else D.copy() | self[x]
        self.overwrite['reg_df'] = True
        self.overwrite['raw'] |= self.overwrite['reg'] | self.overwrite['adm'] | self.overwrite['flg']
        self.overwrite['term'] |= self.overwrite['raw']
        self.overwrite['raw_df'] |= self.overwrite['term']
        self.overwrite['reg_df'] |= self.overwrite['term']
        self.overwrite['X'] |= self.overwrite['raw_df']
        self.overwrite['Y'] |= self.overwrite['reg_df'] | self.overwrite['X']
        
        try:
            self.__dict__ = read(self.rslt).__dict__ | self.__dict__
        except:
            pass

        for k, v in self.overwrite.items():
            if v and k in self:
                del self[k]
        for k in ['fill','term','trf_grid','imp_grid','pred']:
            if k not in self:
                # print(k)
                self[k] = dict()

        self.term_codes = [x for x in listify(self.term_codes) if x != self.infer_term]
        self.crse = uniquify(['_total', *listify(self.crse)])
        self.mlt_grp = ['crse','levl_code','styp_code','term_code']
        self.trf_list = cartesian({k: sorted(setify(v), key=str) for k,v in self.trf_grid.items()})
        self.trf_list = [mysort({k:v for k,v in t.items() if v not in ['drop',None,'']}) for t in self.trf_list]
        imp_default = {'iterations':3, 'mmc':0, 'mmf':'mean_match_default', 'datasets':5, 'tune':True}
        self.imp_list = cartesian(self.imp_grid)
        self.imp_list = [mysort(imp_default | v) for v in self.imp_list]
        self.params_list = sorted([mysort({'imp':imp, 'trf':trf}) for trf, imp in it.product(self.trf_list,self.imp_list)], key=str)


    def get_terms(self):
        opts = {x:self[x] for x in ['cycle_day','overwrite','show']}
        for nm in uniquify([*self.term_codes,self.infer_term]):
            if nm not in self.term:
                print(f'get {nm}')
                self.term[nm] = TERM(term_code=nm, **opts).get_raw()


    def preprocess(self):
        def get(nm):
            if nm in self:
                return False
            print(f'get {nm}')
            return True

        if get('raw_df') or get('reg_df'):
            self.get_terms()

        if get('raw_df'):
            with warnings.catch_warnings(action='ignore'):
                self.raw_df = pd.concat([term.raw for term in self.term.values()], ignore_index=True).dropna(axis=1, how='all').prep()

        if get('reg_df'):
            with warnings.catch_warnings(action='ignore'):
                self.reg_df = {k: pd.concat([term.reg[k].query(f"crse in {self.crse}") for term in self.term.values()]).prep() for k in ['cur','end']}

        where = lambda x: x.query("levl_code == 'ug' and styp_code in ('n','r','t')").copy()
        if get('X'):
            R = self.raw_df.copy()
            repl = {'ae':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'r1':1, 'r2':2, 'r3':3, 'r4':4}
            R['hs_qrtl'] = pd.cut(R['hs_pctl'], bins=[-1,25,50,75,90,101], labels=[4,3,2,1,0], right=False).combine_first(R['apdc_code'].map(repl))
            R['remote'] = R['camp_code'] != 's'
            R['resd'] = R['resd_code'] == 'r'
            R['lgcy'] = ~R['lgcy_code'].isin(['n','o'])
            R['majr_code'] = R['majr_code'].replace({'0000':'und', 'eled':'eted', 'agri':'unda'})
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
            self.X = X.prep().prep_bool().set_index(self.attr, drop=False).rename(columns=lambda x:'__'+x)
            self.X.missing().disp(100)

        if get('Y'):
            Y = {k: self.X[[]].join(y.set_index(['pidm','term_code','crse'])['credit_hr']) for k, y in self.reg_df.items()}
            agg = lambda y: where(y).groupby(self.mlt_grp)['credit_hr'].agg(lambda x: (x>0).sum())
            A = agg(self.reg_df['end'])
            B = agg(Y['end'])
            M = (A / B).replace(np.inf, pd.NA).rename('mlt').reset_index().query(f"term_code != {self.infer_term}").prep()
            N = M.assign(term_code=self.infer_term)
            self.mlt = pd.concat([M, N], axis=0).set_index(self.mlt_grp)
            Y = {k: y.squeeze().unstack().dropna(how='all', axis=1).fillna(0) for k, y in Y.items()}
            self.Y = Y['cur'].rename(columns=lambda x:x+'_cur').join(Y['end']>0).prep()
        self.dump()


    def predict(self, crse, styp_code, params, train_term):
        print(ljust(crse,8), train_term, styp_code, 'creating')
        X = self.X.copy()
        if styp_code != 'all':
            X = X.query(f"styp_code=='{styp_code}'")
        trf = ColumnTransformer([(c,t,["__"+c]) for c,t in params['trf'].items()], remainder='drop', verbose_feature_names_out=False)
        cols = uniquify(['_total_cur',crse+'_cur',crse])
        Z = trf.fit_transform(X).join(self.Y[cols]).prep().prep_bool().prep_category().sort_index()
        y = Z[crse].copy().rename('true').to_frame()
        Z.loc[Z.eval(f"term_code!={train_term}"), crse] = pd.NA

        iterations = params['imp'].pop('iterations')
        datasets = params['imp'].pop('datasets')
        tune = params['imp'].pop('tune')
        mmc = params['imp'].pop('mmc')
        mmf = params['imp'].pop('mmf')
        if mmc > 0 and mmf is not None:
            params['imp']['mean_match_scheme'] = getattr(mf, mmf).copy()
            params['imp']['mean_match_scheme'].set_mean_match_candidates(mmc)
        
        if tune:
            # print('tuning')
            imp = mf.ImputationKernel(Z, datasets=1, **params['imp'])
            imp.mice(iterations=1)
            optimal_parameters, losses = imp.tune_parameters(dataset=0, optimization_steps=5)
        else:
            # print('not tuning')
            optimal_parameters = None
        imp = mf.ImputationKernel(Z, datasets=datasets, **params['imp'])
        imp.mice(iterations=iterations, variable_parameters=optimal_parameters)
        if self.inspect:
            imp.inspect()

        Z.loc[:, crse] = pd.NA
        P = imp.impute_new_data(Z)
        details = pd.concat([y
                .assign(pred=P.complete_data(k)[crse], train_term=train_term, crse=crse, sim=k)
                .set_index(['train_term','crse','sim'], append=True)
            for k in range(P.dataset_count())]).prep_bool()
        agg = lambda x: pd.Series({
            'pred': x['pred'].sum(min_count=1),
            'true': x['true'].sum(min_count=1),
            'mse_pct': ((1*x['pred'] - x['true'])**2).mean()*100,
            'f1_inv_pct': (1-f1_score(x.dropna()['true'], x.dropna()['pred'], zero_division=np.nan))*100,
        })
        summary = details.groupby([*self.mlt_grp,'train_term','sim']).apply(agg).join(self.mlt).rename_axis(index={'term_code':'pred_term'})
        for x in ['pred','true']:
            summary[x] = summary[x] * summary['mlt']
        summary.insert(2, 'err', summary['pred'] - summary['true'])
        summary.insert(3, 'err_pct', (summary['err'] / summary['true']).clip(-1, 1) * 100)
        S = {'details':details, 'summary':summary.drop(columns='mlt').prep()}#, 'trf':trf, 'imp':imp}
        # S['summary'].disp(5)
        return S
        # return S, True


    def analyze(self, df):
        def pivot(df, val):
            Y = (
                df
                .reset_index()
                .pivot_table(columns='train_term', index=['crse','styp_code','pred_term'], values=val, aggfunc=['count',pctl(0),pctl(25),pctl(50),pctl(75),pctl(100)])
                .rename_axis(columns=[val,'train_term'])
                .stack(0, future_stack=True)
                .assign(abs_mean = lambda x: x.abs().mean(axis=1))
            )
            return Y
        mask = df.eval(f"pred_term!={self.infer_term}")
        return {stat: pivot(df[mask], stat) for stat in ["pred","err","err_pct","mse_pct","f1_inv_pct"]} | {"proj": pivot(df[~mask], "pred")}


    def main(self, styp_codes=('n','t','r')):
        self.preprocess()
        styp_codes = listify(styp_codes)
        g = lambda Y: Y | {k: pd.concat([y[k] for y in Y.values() if isinstance(y, dict) and k in y.keys()]).sort_index() for k in ['details','summary']}
        L = len(self.crse) * len(styp_codes) * len(self.params_list)
        k = 0
        start_time = time.perf_counter()
        for crse in self.crse:
            for styp_code in listify(styp_codes):
                for params in self.params_list:
                    for train_term in self.term_codes:
                        path = [crse,styp_code,str(params),train_term]
                        new = False
                        try:
                            y = nest(path, self.pred)
                        except:
                            if not new:
                                print(str(params))
                            y = self.predict(crse, styp_code, copy.deepcopy(params), train_term)
                            nest(path, self.pred, y)
                            new = True
                            self.dump()
                    path.pop(-1)
                    Y = nest(path, self.pred)
                    # return path
                    
                    # return Y
                    if new:
                        for k in ['details', 'summary']:
                            Y[k] = pd.concat([y[k] for y in Y.values() if isinstance(y, dict) and k in y.keys()]).sort_index()
                        Y['rslt'] = self.analyze(Y['summary'])
                        self.dump()
                        k += 1
                    else:
                        L -= 1
                    print(path)
                    Y['rslt']['err_pct'].query("err_pct == ' 50%'").disp(100)
                    Y['summary'].query(f"pred_term!=train_term & pred_term!={self.infer_term}")["err_pct"].abs().describe().to_frame().T.disp(200)
                    self.dump()
                    elapsed = (time.perf_counter() - start_time) / 60
                    complete = k / L if L > 0 else 1
                    rate = elapsed / k if k > 0 else 0
                    remaining = rate * (L - k)
                    print(f"{k} / {L} = {round(complete*100,1)}% complete, elapsed = {round(elapsed,1)} min, remaining = {round(remaining,1)} min @ {round(rate,1)} min per model")
            print("\n========================================================================================================\n")


    # def main(self, styp_codes=('n','t','r')):
    #     self.preprocess()
    #     g = lambda Y: {k: pd.concat([y[k] for y in Y.values() if isinstance(y, dict) and k in y.keys()]).sort_index() for k in ['details','summary']}
    #     start_time = time.perf_counter()
    #     L = len(self.params_list)
    #     k = 0
    #     for params in self.params_list:
    #         # print(str(params))
    #         new = False
    #         Y = []
    #         for crse in self.crse:
    #             for train_term in self.term_codes:
    #                 for styp_code in listify(styp_codes):
    #                     path = [str(params),crse,train_term,styp_code]
    #                     try:
    #                         y = nest(path, self.pred)
    #                     except:
    #                         if not new:
    #                             print(str(params))
    #                         y = self.predict(copy.deepcopy(params), crse, train_term, styp_code)
    #                         nest(path, self.pred, y)
    #                         new = True
    #                         self.dump()
    #                     Y.append(y)
    #         P = self.pred[str(params)]
    #         if new:
    #             for key in ['details', 'summary']:
    #                 P[key] = pd.concat([y[key] for y in Y])
    #             P['rslt'] = self.analyze(P['summary'])
    #             self.dump()
    #             k += 1
    #         else:
    #             L -= 1
    #         P['rslt']['err_pct'].query("err_pct == ' 50%'").disp(100)
    #         P['summary'].query(f"train_term==202308 & pred_term!=202408")["err_pct"].abs().describe().to_frame().T.disp(200)
    #         elapsed = (time.perf_counter() - start_time) / 60
    #         complete = k / L if L > 0 else 1
    #         rate = elapsed / k if k > 0 else 0
    #         remaining = rate * (L - k)
    #         print(f"{k} / {L} = {round(complete*100,1)}% complete, elapsed = {round(elapsed,1)} min, remaining = {round(remaining,1)} min @ {round(rate,1)} min per model")
    #         print("\n========================================================================================================\n")
    

code_desc = lambda x: [x+'_code', x+'_desc']
passthru = ['passthrough']
passdrop = ['passthrough', 'drop']
# passthru = passdrop
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
        'birth_day': ['median',['term_code','styp_code']],
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
        *code_desc('term'),
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
    # 'cycle_day': (TERM(term_code=202408).cycle_date-pd.Timestamp.now()).days+1,
    'cycle_day': 183,
    'crse': [
        'engl1301',
        # 'biol1406',
        # 'math1314',
        # 'biol2401',
        # 'math2412',
        # 'agri1419',
        # 'psyc2301',
        # 'ansc1319',
        # 'comm1311',
        # 'hist1301',
        # 'govt2306',
        # 'math1324',
        # 'chem1411',
        # 'univ0301',
        # 'univ0204',
        # 'univ0304',
        # 'agri1100',
        # 'comm1315',
        # 'agec2317',
        # 'govt2305',
        # 'busi1301',
        # 'arts1301',
        # 'math1342',
        # 'math2413',
        ],
    'trf_grid': {
        'act_equiv': passthru,
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
        'waiver': passdrop,
        'writing': passthru,
        },
    'imp_grid': {
        'mmc': 10,
        # 'datasets': 25,
        'datasets': 1,
        'iterations': 1,
        'tune': False,
    },
    'overwrite': {
        # # 'trm':True,
        # 'reg':True,
        # 'adm':True,
        # 'flg':True,
        # 'raw':True,
        # 'term': True,
        # 'raw_df': True,
        # 'reg_df': True,
        # 'X': True,
        # 'Y': True,
        # 'pred': True,
    },
    # 'inspect': True,
}


if __name__ == "__main__":
    from IPython.utils.io import Tee
    self = AMP(**kwargs)
    with contextlib.closing(Tee(self.rslt.with_suffix('.txt'), "w", channel="stdout")) as outputstream:
        print(pd.Timestamp.now())
        self.preprocess()
        # self.params_list = self.params_list[102:103]
        self.main(styp_codes='n')
        # print(len(self.params_list))
        # for x in self.params_list:
        #     print(x)
        # T = TERM(202008, cycle_day=184, show={'adm':True}).get_adm(184)