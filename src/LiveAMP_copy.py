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

    def dump(self):
        return write(self.rslt, self, overwrite=True)

    def __post_init__(self):
        self.rslt = root_path / f"resources/rslt/{rjust(self.cycle_day,3,0)}/rslt.pkl"
        mkdir(self.rslt.parent)
        D = {'trm':False, 'adm':False, 'reg':False, 'flg':False, 'raw':False, 'term':False, 'raw_df':False, 'reg_df':False, 'X':False, 'Y':False,
             'transformed':False, 'imputed':False, 'predicted':False, 'performance':False, 'optimal_params':False, 'optimal_predicted':False, 'summary':False}
        
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
        # for k in ['fill','term','trf_grid','imp_grid','transformed','imputed','predicted','summary']:
        for k in ['fill','trf_grid','imp_grid']:
            if k not in self:
                # print(k)
                self[k] = dict()

        self.crse = uniquify(['_total', *listify(self.crse)])
        self.styp_codes = uniquify(self.styp_codes)
        self.term_codes = [x for x in uniquify(self.term_codes) if x != self.infer_term]
        self.mlt_grp = ['crse','levl_code','styp_code','term_code']

        self.trf_list = cartesian({k: uniquify(v, key=str) for k,v in self.trf_grid.items()})
        self.trf_list = [uniquify({k:v for k,v in t.items() if v not in ['drop',None,'']}) for t in self.trf_list]
        
        imp_default = {'datasets':10, 'iterations':3, 'mmc':10, 'mmf':'mean_match_default', 'tune':True}
        self.imp_list = cartesian(self.imp_grid)
        self.imp_list = [uniquify(imp_default|v) for v in self.imp_list]
        
        clf_default = {'datasets':10, 'iterations':3, 'mmc':10, 'mmf':'mean_match_default', 'tune':True}
        self.clf_list = cartesian(self.clf_grid)
        self.clf_list = [uniquify(clf_default | v) for v in self.clf_list]
        self.params_list = mysort([uniquify({'clf':clf, 'imp':imp, 'trf':trf}) for clf, imp, trf in it.product(self.clf_list, self.imp_list, self.trf_list)], key=str)


    def get_terms(self):
        opts = {x:self[x] for x in ['cycle_day','overwrite','show']}
        for term_code in uniquify([*self.term_codes,self.infer_term]):
            A, b, path = self.get(['term',term_code])
            if b:
                self.set(path, TERM(term_code=term_code, **opts).get_raw())


    # def impute(self):
    #     for styp_code in ['n','t','r']:
    #         for trf_spec in self.trf_list():
    #             trf = ColumnTransformer([(c,t,["__"+c]) for c,t in trf_spec.items()], remainder='passthrough', verbose_feature_names_out=False)
    #             X = self.X.copy()
    #             if styp_code != 'all':
    #                 X = X.query(f"styp_code=='{styp_code}'")
    #             X = trf.fit_transform(X).prep().prep_bool().prep_category().sort_index()
    #             self.train(X, )

    #     trf = ColumnTransformer([(c,t,["__"+c]) for c,t in params['trf'].items()], remainder='drop', verbose_feature_names_out=False)
    #     cols = uniquify(['_total_cur',crse+'_cur',crse])




    # def predict(self, crse, styp_code, params, train_term):
    #     print(ljust(crse,8), styp_code, train_term, 'creating')
    #     X = self.X.copy()
    #     if styp_code != 'all':
    #         X = X.query(f"styp_code=='{styp_code}'")
    #     trf = ColumnTransformer([(c,t,["__"+c]) for c,t in params['trf'].items()], remainder='pasthrough', verbose_feature_names_out=False)
    #     cols = uniquify(['_total_cur',crse+'_cur',crse])
    #     Z = trf.fit_transform(X).join(self.Y[cols]).prep().prep_bool().prep_category().sort_index()
    #     y = Z[crse].copy().rename('actual').to_frame()
    #     Z.loc[Z.eval(f"term_code!={train_term}"), crse] = pd.NA

    #     iterations = params['imp'].pop('iterations')
    #     datasets = params['imp'].pop('datasets')
    #     tune = params['imp'].pop('tune')
    #     mmc = params['imp'].pop('mmc')
    #     mmf = params['imp'].pop('mmf')
    #     if mmc > 0 and mmf is not None:
    #         params['imp']['mean_match_scheme'] = getattr(mf, mmf).copy()
    #         params['imp']['mean_match_scheme'].set_mean_match_candidates(mmc)
        
    #     if tune:
    #         # print('tuning')
    #         imp = mf.ImputationKernel(Z, datasets=1, **params['imp'])
    #         imp.mice(iterations=1)
    #         optimal_parameters, losses = imp.tune_parameters(dataset=0, optimization_steps=5)
    #     else:
    #         # print('not tuning')
    #         optimal_parameters = None
    #     imp = mf.ImputationKernel(Z, datasets=datasets, **params['imp'])
    #     imp.mice(iterations=iterations, variable_parameters=optimal_parameters)
    #     if self.inspect:
    #         imp.inspect()

    #     Z.loc[:, crse] = pd.NA
    #     P = imp.impute_new_data(Z)
    #     details = pd.concat([y
    #             .assign(predict=P.complete_data(k)[crse], train_term=train_term, crse=crse, sim=k)
    #             .set_index(['train_term','crse','sim'], append=True)
    #         for k in range(P.dataset_count())]).prep_bool()
    #     return details
    
    # def aggregate(self, details):
    #     agg = lambda x: pd.Series({
    #         'predict': x['predict'].sum(min_count=1),
    #         'actual': x['actual'].sum(min_count=1),
    #         'mse_pct': ((1*x['predict'] - x['actual'])**2).mean()*100,
    #         'f1_inv_pct': (1-f1_score(x.dropna()['actual'], x.dropna()['predict'], zero_division=np.nan))*100,
    #     })
    #     summary = details.groupby([*self.mlt_grp,'train_term','sim']).apply(agg).join(self.mlt)#.rename_axis(index={'term_code':'pred_term'})
    #     for x in ['predict','actual']:
    #         summary[x] = summary[x] * summary['mlt']
    #     summary.insert(2, 'error', summary['predict'] - summary['actual'])
    #     summary.insert(3, 'error_pct', summary['error'] / summary['actual'] * 100)
    #     return summary
    #     # S = {'details':details, 'summary':summary.drop(columns='mlt').prep()}#, 'trf':trf, 'imp':imp}
    #     # S['summary'].disp(5)
    #     # return S
    #     # return S, True


    # def analyze(self, df):
    #     def pivot(df, val):
    #         Y = (
    #             df
    #             .query(f"term_code!=train_term")
    #             .reset_index()
    #             .pivot_table(columns='train_term', index=['crse','styp_code','term_code'], values=val, aggfunc=['count',pctl(0),pctl(25),pctl(50),pctl(75),pctl(100)])
    #             .rename_axis(columns=[val,'train_term'])
    #             .stack(0, future_stack=True)
    #             .assign(abs_mean = lambda x: x.abs().mean(axis=1))
    #         )
    #         return Y
    #     mask = df.eval(f"term_code!={self.infer_term}")
    #     return {stat: pivot(df[mask], stat) for stat in ["predict","error","error_pct","mse_pct","f1_inv_pct"]} | {"project": pivot(df[~mask], "predict")}


    # def main(self, styp_codes=('n','t','r')):
    #     self.preprocess()
    #     styp_codes = listify(styp_codes)
    #     g = lambda Y: Y | {k: pd.concat([y[k] for y in Y.values() if isinstance(y, dict) and k in y.keys()]).sort_index() for k in ['details','summary']}
    #     L = len(self.crse) * len(styp_codes) * len(self.params_list)
    #     k = 0
    #     start_time = time.perf_counter()
    #     self.optimal = dict()
    #     for crse in self.crse:
    #         for styp_code in listify(styp_codes):
    #             for params_idx, params in enumerate(self.params_list):
    #                 print("\n========================================================================================================\n")
    #                 print(ljust(crse,8),styp_code,params_idx)
    #                 new = False
    #                 for train_term in self.term_codes:
    #                     path = [crse,styp_code,str(params),train_term,'details']
    #                     try:
    #                         details = nest(path, self.pred)
    #                     except:
    #                         new = True
    #                         nest(path[:-1], self.pred, dict())
    #                         details = self.predict(crse, styp_code, copy.deepcopy(params), train_term)
    #                         nest(path, self.pred, details)
    #                     path[-1] = 'summary'
    #                     try:
    #                         summary = nest(path, self.pred)
    #                     except:
    #                         summary = self.aggregate(details)
    #                         nest(path, self.pred, summary)
    #                         self.dump()
    #                 Y = nest(path[:-2], self.pred)
    #                 for key in ['details', 'summary']:
    #                     Y[key] = pd.concat([y[key] for y in Y.values() if isinstance(y, dict) and key in y.keys()]).sort_index()
    #                 Y['rslt'] = self.analyze(Y['summary'])
    #                 if new:
    #                     # self.dump()
    #                     k += 1
    #                 else:
    #                     L -= 1
    #                 # Y['rslt']['error_pct'].query("error_pct == ' 50%'").round(decimals=2).disp(100)
    #                 E = Y['summary'].query(f"term_code!=train_term & term_code!={self.infer_term}")["error_pct"].abs()
    #                 # E.describe().to_frame().T.round(decimals=2).disp(200)
    #                 new = Y | {'params_idx':params_idx, 'params':params, 'score':E.median()}
    #                 print(f"new score = {new['score']:.2f}")
    #                 if pd.notnull(new['score']) and new['score'] < 30:
    #                     try:
    #                         old = nest(path[:-3], self.optimal)
    #                         print(f"old score = {old['score']:.2f}")
    #                         if new['score'] < old['score']:
    #                             print('replacing')
    #                             nest(path[:-3], self.optimal, new)
    #                         else:
    #                             print('keeping')
    #                     except:
    #                         nest(path[:-3], self.optimal, new)
    #                 elapsed = (time.perf_counter() - start_time) / 60
    #                 complete = k / L if L > 0 else 1
    #                 rate = elapsed / k if k > 0 else 0
    #                 remaining = rate * (L - k)
    #                 print(f"{k} / {L} = {complete*100:.2f}% complete, elapsed = {elapsed:.2f} min, remaining = {remaining:.2f} min @ {rate:.2f} min per model")
    #         self.dump()
    #     self.push()


    def preprocess(self):
        if self.get('raw_df')[1] or self.get('reg_df')[1]:
            self.get_terms()
        
        A, b, path = self.get('raw_df')
        if b:
            with warnings.catch_warnings(action='ignore'):
                self.set(path, pd.concat([term.raw for term in self.term.values()], ignore_index=True).dropna(axis=1, how='all').prep())
        
        A, b, path = self.get('reg_df')
        if b:
            with warnings.catch_warnings(action='ignore'):
                self.set(path, {k: pd.concat([term.reg[k].query(f"crse in {self.crse}") for term in self.term.values()]).prep() for k in ['cur','end']})

        where = lambda x: x.query("levl_code == 'ug' and styp_code in ('n','r','t')").copy()
        A, b, path = self.get('X')
        if b:
            R = self.raw_df.copy()
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
            self.set(path, X.join(M).prep().prep_bool().set_index(self.attr, drop=False).rename(columns=lambda x:'__'+x))
            
        A, b, path = self.get('Y')
        if b:
            Y = {k: self.X[[]].join(y.set_index(['pidm','term_code','crse'])['credit_hr']) for k, y in self.reg_df.items()}
            agg = lambda y: where(y).groupby(self.mlt_grp)['credit_hr'].agg(lambda x: (x>0).sum())
            A = agg(self.reg_df['end'])
            B = agg(Y['end'])
            M = (A / B).replace(np.inf, pd.NA).rename('mlt').reset_index().query(f"term_code != {self.infer_term}").prep()
            M['mlt_term'] = M['term_code']
            N = M.copy().assign(term_code=self.infer_term)
            self.mlt = pd.concat([M, N], axis=0).set_index([*self.mlt_grp,'mlt_term'])
            Y = {k: y.squeeze().unstack().dropna(how='all', axis=1).fillna(0) for k, y in Y.items()}
            self.set(path, Y['cur'].rename(columns=lambda x:x+'_cur').join(Y['end']>0).prep())
        self.dump()


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
    
    def get(self, path, params=None):
        if params is not None:
            path['params'] = str(params)
        try:
            return nest(path, self.__dict__), False, path
        except:
            return None, True, path

    def set(self, path, val):
        nest(path, self.__dict__, val)
        print(f'created {path}')
        self.dump()


    def get_transformed(self, path, params=None):
        params = None if params is None else subdct(params, ['trf'], True)
        A, b, path = self.get(path | {'nm':'transformed', 'crse':'_total', 'train_term':'all'}, params)
        if b:
            self.preprocess()
            trf = ColumnTransformer([(c,t,["__"+c]) for c,t in params['trf'].items()], remainder='drop', verbose_feature_names_out=False)
            qry = "pidm.notnull()"
            if path['styp_code'] != 'all':
                qry += f" & styp_code == @path['styp_code']"
            if path['train_term'] != 'all':
                qry += f" & term_code == @path['train_term']"
            A = trf.fit_transform(self.X.query(qry)).prep().prep_bool().prep_category().sort_index()
            self.set(path, A)
        return A, b


    def get_imputed(self, path, params=None):
        params = None if params is None else subdct(params, ['imp', 'trf'], True)
        A, b, path = self.get(path | {'nm':'imputed', 'crse':'_total', 'train_term':'all'}, params)
        if b:
            T = self.get_transformed(path, params)[0]
            imp = self.get_model(T, params['imp'])
            A = pd.concat([imp.complete_data(k).assign(imp=k).set_index('imp', append=True) for k in range(imp.dataset_count())])
            self.set(path, A)
        return A, b


    def get_predicted(self, path, params=None):
        A, b, path = self.get(path | {'nm':'predicted'}, params)
        if b:
            I = self.get_imputed(path, params)[0]
            cols = uniquify(['_total_cur', path['crse']+'_cur', path['crse']], False)
            Z = I.join(self.Y[cols]).prep().prep_bool().prep_category().sort_index()
            actual = Z[path['crse']].copy().rename('actual').to_frame()
            Z.loc[Z.eval(f"term_code!=@path['train_term']"), path['crse']] = pd.NA
            clf = self.get_model(Z, params['clf'])
            Z.loc[:, path['crse']] = pd.NA
            predicted = clf.impute_new_data(Z)
            A = pd.concat([actual.assign(
                        predicted = predicted.complete_data(k)[path['crse']],
                        crse = path['crse'],
                        train_term = path['train_term'],
                        sim = k)
                    .set_index(['crse', 'train_term', 'sim'], append=True)
                for k in range(predicted.dataset_count())]).prep_bool()
            self.set(path, A)
        return A, b

    # def get_summary(self, path, params=None):
    #     A, b, path = self.get(path | {'nm':'summary'}, params)
    #     if b:
    #         P = self.get_predicted(path, params)[0]
    #         agg = lambda x: pd.Series({
    #             'predicted': x['predicted'].sum(min_count=1),
    #             'actual': x['actual'].sum(min_count=1),
    #             'mse_pct': ((1*x['predicted'] - x['actual'])**2).mean()*100,
    #             'f1_inv_pct': (1-f1_score(x.dropna()['actual'], x.dropna()['predicted'], zero_division=np.nan))*100,
    #         })
    #         A = P.groupby([*self.mlt_grp,'train_term','imp','sim']).apply(agg).join(self.mlt)#.rename_axis(index={'term_code':'pred_term'})
    #         for x in ['predicted','actual']:
    #             A[x] = A[x] * A['mlt']
    #         A.insert(2, 'error', A['predicted'] - A['actual'])
    #         A.insert(3, 'error_pct', A['error'] / A['actual'] * 100)
    #         self.set(path, A)
    #     return A, b

    def get_performance(self, path, params=None):
        A, b, path = self.get(path | {'nm':'performance'}, params)
        if b:
            P = self.get_predicted(path, params)[0]
            A = 100*(P['actual'] == P['predicted']).mean()
            self.set(path, A)
        return A, b
    
    def get_optimal_params(self, path, params=None):
        params = None
        A, b, path = self.get(path | {'nm':'optimal_params'}, params)
        if b:
            P = self.get_performance(path, params)[0]
            A = min(P, key=P.get)
            self.set(path, A)
        return A, b

    def get_optimal_predicted(self, path, params=None):
        params = None
        A, b, path = self.get(path | {'nm':'optimal_predicted'}, params)
        if b:
            params = self.get_optimal_params(path, params)[0]
            A = self.get_predicted(path, params)[0]
            self.set(path, A)
        return A, b

    def get_summary(self, path, params=None):
        params = None
        A, b, path = self.get(path | {'nm':'summary'}, params)
        if b:
            P = self.get_optimal_predicted(path, params)[0]
            agg = lambda x: pd.Series({
                'predicted': x['predicted'].sum(min_count=1),
                'actual': x['actual'].sum(min_count=1),
                'mse_pct': ((1*x['predicted'] - x['actual'])**2).mean()*100,
                'f1_inv_pct': (1-f1_score(x.dropna()['actual'], x.dropna()['predicted'], zero_division=np.nan))*100,
            })
            A = P.groupby([*self.mlt_grp,'train_term','imp','sim']).apply(agg).join(self.mlt)#.rename_axis(index={'term_code':'pred_term'})
            for x in ['predicted','actual']:
                A[x] = A[x] * A['mlt']
            A.insert(2, 'error', A['predicted'] - A['actual'])
            A.insert(3, 'error_pct', A['error'] / A['actual'] * 100)
            self.set(path, A)
        return A, b



    def main(self):
        self.preprocess()
        for nm in ['transformed', 'imputed', 'predicted', 'performance', 'optimal_params', 'optimal_predicted', 'summary']:
            progress = [len(self.crse) * len(self.styp_codes) * len(self.term_codes) * len(self.params_list), 0]
            start_time = time.perf_counter()
            w = 100
            print("=" * w)
            print(nm)
            for crse in self.crse:
                for styp_code in self.styp_codes:
                    for train_term in self.term_codes:
                        # path = {'nm':nm, 'crse':crse, 'styp_code':styp_code, 'train_term':train_term}
                        for params in self.params_list:
                            # path = {'nm':nm, 'crse':crse, 'styp_code':styp_code, 'train_term':train_term, 'params':json.dumps(mysort(params))}
                            path = {'nm':nm, 'crse':crse, 'styp_code':styp_code, 'train_term':train_term}
                            # print(path)
                            c = getattr(self, 'get_'+nm)(path, params)[1]
                            progress[c] += (2*c-1)
                            if c:
                                elapsed = (time.perf_counter() - start_time) / 60
                                complete = progress[1] / progress[0] if progress[0] > 0 else 1
                                rate = elapsed / progress[1] if progress[1] > 0 else 0
                                remaining = rate * (progress[0] - progress[1])
                                print(f"{progress[1]} / {progress[0]} = {complete*100:.2f}% complete, elapsed = {elapsed:.2f} min, remaining = {remaining:.2f} min @ {rate:.2f} min per model")
                                print("-" * w)
                self.dump()
        # for nm in ['optimal']:


    # def combine(self):
    #     # for key in ['details', 'summary']:
    #     key = 'summary'
    #     A = pd.concat([S[key] for crse, C in self.optimal.items() for styp_code, S in C.items() if isinstance(S, dict) and key in S.keys()])
    #     if key == 'summary':
    #         B = A.copy().reset_index().assign(styp_code=A.reset_index()['styp_code'].replace({'n':'new first time','t':'transfer','r':'returning'}))
    #         C = B.assign(styp_code='all').groupby(A.index.names)[['predict','actual','error']].sum().reset_index()
    #         C['error_pct'] = C['error'] / C['actual'] * 100
    #         A = pd.concat([B,C])
    #     self.optimal[key] = A
    #     write(self[key], self.optimal[key], index=False)
    #     self.dump()

    # def push(self):
    #     self.combine()
    #     target_url = 'https://prod-121.westus.logic.azure.com:443/workflows/784fef9d36024a6abf605d1376865784/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=1Yrr4tE1SwYZ88SU9_ixG-WEdN1GFicqJwH_KiCZ70M'
    #     with open(self.summary, 'rb') as target_file:
    #         response = requests.post(target_url, files = {"amp_summary.csv": target_file})
    #     print('file pushed')

code_desc = lambda x: [x+'_code', x+'_desc']
passthru = ['passthrough']
passdrop = ['passthrough', 'drop']
# passdrop = passthru
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
    'cycle_day': (TERM(term_code=202408).cycle_date-pd.Timestamp.now()).days+1,
    # 'cycle_day': 176,
    'crse': [
        # 'agec2317',
        # 'agri1100',
        # 'agri1419',
        # 'ansc1319',
        # 'arts1301',
        # 'biol1406',
        # 'biol2401',
        # 'busi1301',
        # 'comm1311',
        # 'comm1315',
        # 'engl1301',
        # 'govt2305',
        # 'govt2306',
        # 'hist1301',
        # 'math1314',
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
        'waiver': passdrop,
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
        # 'transformed': True,
        # 'imputed': True,
        # 'predicted': True,
        # 'performance': True,
        # 'optimal_params': True,
        # 'optimal_predicted': True,
        'summary': True,
    },
    'styp_codes': ['n','t','r'],
}


if __name__ == "__main__":
    @pd_ext
    def disp(df, max_rows=4, max_cols=200, **kwargs):
        display(HTML(df.to_html(max_rows=max_rows, max_cols=max_cols, **kwargs)))
        print(df.head(max_rows).reset_index().to_markdown(tablefmt='psql'))

    from IPython.utils.io import Tee
    kwargs['styp_codes'] = ['n','t']
    self = AMP(**kwargs)
    with contextlib.closing(Tee(self.rslt.with_suffix('.txt'), "w", channel="stdout")) as outputstream:
        print(pd.Timestamp.now())
        self.main()
        # self.push()