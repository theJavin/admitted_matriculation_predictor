from term import *
import requests, miceforest as mf
from sklearn.compose import make_column_selector, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, KBinsDiscretizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn import set_config
set_config(transform_output="pandas")

stringify = lambda x: str(x).replace('\n','').replace(' ','')

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
    crse_codes: typing.List
    attr: typing.List
    styp_codes: tuple = ('n','t','t')
    fill: typing.Dict = None
    trf_grid: typing.Dict = None
    imp_grid: typing.Dict = None
    clf_grid: typing.Dict = None
    overwrite: typing.Dict = None
    show: typing.Dict = None

            # A = {'str':T, 'par':[x for x in self.clf_imp_trf_list if stringify(x)==T][0]}
            # idx = [stringify(x) for x in self.clf_imp_trf_list].index(T)
            # A = {'idx':idx, 'str':T, 'par':self.clf_imp_trf_list[idx]}
        #     self.get(path, A)
        # return A, b, path

    # def get_performance(self):
    #     grid = {'nm':'performance', 'styp_code':self.styp_codes, 'train_code':self.term_codes, 'crse_code':self.crse_codes}
    #     def func(path, par):
    #         P = self.get(path | {'nm':'predicted', 'par':stringify(par)}).join(self.mlt['all']).reset_index()
    #         T = P[['pred_code','pred_desc']].drop_duplicates()
    #         for k in ['train','mlt']:
    #             P = P.merge(T.rename(columns=lambda x: x.replace('pred',k)))
    #         for k in ['predicted','actual']:
    #             P[k+'_mlt'] = P[k] * P['mlt']
    #         Q = P.copy().assign(styp_code='all', styp_desc='all incoming')
    #         return (
    #             pd.concat([P,Q])
    #             .groupby(['crse_code','levl_code','levl_desc','styp_code','styp_desc','pred_code','pred_desc','train_code','train_desc','mlt_code','mlt_desc','imp','sim'])
    #             .apply(lambda x: pd.Series({
    #                 'predicted': x['predicted_mlt'].sum(),
    #                 'actual': x['actual_mlt'].sum(),
    #                 'error': x['predicted_mlt'].sum() - x['actual_mlt'].sum(),
    #                 'error_pct': (x['predicted_mlt'].sum() - x['actual_mlt'].sum()) / x['actual_mlt'].sum() * 100,
    #                 # 'acc_pct': (x['predicted'] == x['actual']).mean()*100,
    #                 'acc_pct': accuracy_score(x['actual'], x['predicted'])*100,
    #                 'bal_acc_pct': balanced_accuracy_score(x['actual'], x['predicted'])*100,
    #                 'f1_pct': f1_score(x['actual'], x['predicted'], zero_division=np.nan)*100,
    #                 'mlt': x['mlt'].mean(),
    #                 # 'zero': x['mlt'].var(),
    #                 # 'mse_pct': ((1*x['predicted'] - x['actual'])**2).mean()*100,
    #                 # 'f1_inv_pct': (1-f1_score(x['actual'], x['predicted'], zero_division=np.nan))*100,
    #             }), include_groups=False)
    #         )
    #     self.run(grid, self.loop_par(self.clf_imp_trf_list, func))
    #         # P = self.get_predicted(path)[0][p]
    #         # A[p] = 100*(P['predicted'] == P['actual']).mean()


    # def get_optimal(self, path):
    #     path = path | {'nm':'optimal'}
    #     A, b = self.get(path)
    #     if b:
    #         P = self.get_performance(path)[0]

    #         T = min(P, key=P.get)
    #         A = {'str':T, 'par':[x for x in self.clf_imp_trf_list if stringify(x)==T][0]}
    #         # idx = [stringify(x) for x in self.clf_imp_trf_list].index(T)
    #         # A = {'idx':idx, 'str':T, 'par':self.clf_imp_trf_list[idx]}
    #         self.get(path, A)
    #     return A, b, path


    # def get_details(self, path):
    #     path = path | {'nm':'details'}
    #     A, b = self.get(path)
    #     if b:
    #         p = self.get_optimal(path)[0]['str']
    #         A = self.get_predicted(path)[0][p]
    #         self.get(path, A)
    #     return A, b, path


    # def get_summary(self, path):
    #     path = path | {'nm':'summary'}
    #     A, b = self.get(path)
    #     if b:
    #         D = self.get_details(path)[0].join(self.mlt).reset_index()
    #         T = D[['pred_code','pred_desc']].drop_duplicates()
    #         for k in ['train','mlt']:
    #             D = D.merge(T.rename(columns=lambda x: x.replace('pred',k)))
    #         for k in ['predicted','actual']:
    #             D[k+'_mlt'] = D[k] * D['mlt']
    #         E = D.copy().assign(styp_code='all', styp_desc='all incoming')
    #         A = (
    #             pd.concat([D,E])
    #             .groupby(['crse_code','levl_code','levl_desc','styp_code','styp_desc','pred_code','pred_desc','train_code','train_desc','mlt_code','mlt_desc','imp','sim'])
    #             .apply(lambda x: pd.Series({
    #                 'predicted': x['predicted_mlt'].sum(),
    #                 'actual': x['actual_mlt'].sum(),
    #                 'error': x['predicted_mlt'].sum() - x['actual_mlt'].sum(),
    #                 'error_pct': (x['predicted_mlt'].sum() - x['actual_mlt'].sum()) / x['actual_mlt'].sum() * 100,
    #                 'mse_pct': ((1*x['predicted'] - x['actual'])**2).mean()*100,
    #                 'f1_inv_pct': (1-f1_score(x['actual'], x['predicted'], zero_division=np.nan))*100,
    #             }), include_groups=False)
    #         )
    #         self.get(path, A)
    #     return A, b, path

    # def run(self, nm):
    #     L = cartesian({'nm':'X', 'styp_code':'all', 'crse':'all', 'train_term':'all'})
    #     L = cartesian({'nm':'transformed', 'styp_code':self.styp_codes, 'crse':'all', 'train_term':'all'})
    #     L = cartesian({'nm':'imputed', 'styp_code':self.styp_codes, 'crse':'all', 'train_term':'all'})
    #     L = cartesian({'nm':'predicted', 'styp_code':self.styp_codes, 'crse':self.crse, 'train_term':self.term_codes})
    #     start_time = time.perf_counter()
    #     for k, path in enumerate(L):
    #         A, b, path = getattr(self, 'get_'+nm)(path)

        
    #     L = self.styp_codes
    #     L = list(it.product(self.styp_codes, self.crse, self.term_codes))

    
    
    # def run(self, nm):
    #     self.run(self.dependancy[nm])
    #     progress = [len(self.styp_codes), 0]
    #     if nm not in ['transformed', 'imputed']:
    #         progress[0] *= len(self.crse) * len(self.term_codes)
    #     # progress = [len(self.styp_codes) * len(self.crse) * len(self.term_codes), 0]
    #     start_time = time.perf_counter()
    #     print("=" * 100)
    #     print(nm)
    #     for styp_code in self.styp_codes:
    #         for crse in self.crse:
    #             for train_code in self.term_codes:
    #                 path = {'nm':nm, 'styp_code':styp_code, 'crse':crse, 'train_code':train_code}
    #                 return getattr(self, 'get_'+nm)(path)
    #                 A, b, path = getattr(self, 'get_'+nm)(path)
    #                 progress[1] += b
    #                 # progress[b] += (2*b-1)
    #                 elapsed = (time.perf_counter() - start_time) / 60
    #                 complete = progress[1] / progress[0] if progress[0] > 0 else 1
    #                 rate = elapsed / progress[1] if progress[1] > 0 else 0
    #                 remaining = rate * (progress[0] - progress[1])
    #                 msg = f"{join(path.values())}; complete: {progress[1]} / {progress[0]} = {complete*100:.2f}%; elapsed = {elapsed:.2f} min; remaining = {remaining:.2f} min @ {rate:.2f} min per model"
    #                 if b:
    #                     print(msg)
    # def run(self, nm):
    #     progress = [len(self.crse) * len(self.styp_codes) * len(self.term_codes), 0]
    #     start_time = time.perf_counter()
    #     print("=" * 100)
    #     print(nm)
    #     for crse in self.crse:
    #         for styp_code in self.styp_codes:
    #             for train_code in self.term_codes:
    #                 path = {'nm':nm, 'crse':crse, 'styp_code':styp_code, 'train_code':train_code}
    #                 A, b, path = getattr(self, 'get_'+nm)(path)
    #                 progress[b] += (2*b-1)
    #                 elapsed = (time.perf_counter() - start_time) / 60
    #                 complete = progress[1] / progress[0] if progress[0] > 0 else 1
    #                 rate = elapsed / progress[1] if progress[1] > 0 else 0
    #                 remaining = rate * (progress[0] - progress[1])
    #                 msg = f"{join(path.values())}; complete: {progress[1]} / {progress[0]} = {complete*100:.2f}%; elapsed = {elapsed:.2f} min; remaining = {remaining:.2f} min @ {rate:.2f} min per model"
    #                 if b:
    #                     print(msg)

    # def get_terms(self, path=None):
    #     grid = {'nm':'terms', 'styp_code':'all'}
    #     [getattr(self,'get_'+attr)() for attr in listify(self.dependancy[grid['nm']])]
    #     A = self.get(path)
    #     opts = {x:self[x] for x in ['cycle_day','overwrite','show']}
    #     for term_code in uniquify([*self.term_codes,self.infer_term]):
    #         if term_code not in A:
    #             A[term_code] = TERM(term_code=term_code, **opts).get_raw()
    #     self.get(path, A)
    #     return A
    
    # def get_raw_df(self, path=None):
    #     grid = {'nm':'raw_df', 'styp_code':'all'}
    #     [getattr(self,'get_'+attr)() for attr in listify(self.dependancy[grid['nm']])]
        
    #     A = self.get(path)
    #     if len(A)==0:
    #         grid = {'nm':'raw_df', 'styp_code':'all'}
    #         with warnings.catch_warnings(action='ignore'):
    #             A = self.where(pd.concat([term.raw for term in self.terms['all'].values()], ignore_index=True).dropna(axis=1, how='all')).prep()
    #     self.get(path, A)
    #     return A

    # def get_reg_df(self, path=None):
    #     nm = 'reg_df'
    #     path = {'nm':nm, 'styp_code':'all'}
    #     [getattr(self,'get_'+attr)() for attr in listify(self.dependancy[nm])]
    #     A = self.get(path)
    #     if len(A)==0:
    #         with warnings.catch_warnings(action='ignore'):
    #             A = {k: self.where(pd.concat([term.reg[k].query(f"crse in {self.crse}") for term in self.terms['all'].values()])).prep().set_index(['pidm','pred_code','crse']) for k in ['cur','end']}
    #     self.get(path, A)
    #     return A

    # def get_X(self, path=None):
    #     nm = 'X'
    #     path = {'nm':nm, 'styp_code':'all'}
    #     [getattr(self,'get_'+attr)() for attr in listify(self.dependancy[nm])]
    #     A = self.get(path)
    #     if len(A)==0:
    #         R = self.raw_df['all']
    #         repl = {'ae':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'r1':1, 'r2':2, 'r3':3, 'r4':4}
    #         R['hs_qrtl'] = pd.cut(R['hs_pctl'], bins=[-1,25,50,75,90,101], labels=[4,3,2,1,0], right=False).combine_first(R['apdc_code'].map(repl))
    #         R['remote'] = R['camp_code'] != 's'
    #         R['resd'] = R['resd_code'] == 'r'
    #         R['lgcy'] = ~R['lgcy_code'].isin(['n','o'])
    #         R['majr_code'] = R['majr_code'].replace({'0000':pd.NA, 'und':pd.NA, 'eled':'eted', 'agri':'unda'})
    #         R['coll_code'] = R['coll_code'].replace({'ae':'an', 'eh':'ed', 'hs':'hl', 'st':'sm', '00':pd.NA})
    #         R['coll_desc'] = R['coll_desc'].replace({
    #             'ag & environmental_sciences':'ag & natural_resources',
    #             'education & human development':'education',
    #             'health science & human_service':'health sciences',
    #             'science & technology':'science & mathematics'})
    #         majr = ['majr_desc','dept_code','dept_desc','coll_code','coll_desc']
    #         S = R.sort_values('cycle_date').drop_duplicates(subset='majr_code', keep='last')[['majr_code',*majr]]
    #         X = R.drop(columns=majr).merge(S, on='majr_code', how='left').prep().prep_bool()

    #         checks = [
    #             'cycle_day >= 0',
    #             'apdc_day >= cycle_day',
    #             'appl_day >= apdc_day',
    #             'birth_day >= appl_day',
    #             'birth_day >= 5000',
    #             'distance >= 0',
    #             'hs_pctl >=0',
    #             'hs_pctl <= 100',
    #             'hs_qrtl >= 0',
    #             'hs_qrtl <= 4',
    #             'act_equiv >= 1',
    #             'act_equiv <= 36',
    #             'gap_score >= 0',
    #             'gap_score <= 100',
    #         ]
    #         for check in checks:
    #             mask = X.eval(check)
    #             assert mask.all(), [check,X[~mask].disp(5)]
    #         for k, v in self.fill.items():
    #             X[k] = X.impute(k, *listify(v))
    #         M = X.isnull().rename(columns=lambda x:x+'_missing')
    #         A = X.join(M).prep().prep_bool().set_index(self.attr, drop=False).rename(columns=lambda x:'__'+x)
    #     self.get(path, A)
    #     return A

    # def get_Y(self, path=None):
    #     nm = 'Y'
    #     path = {'nm':nm, 'styp_code':'all'}
    #     [getattr(self,'get_'+attr)() for attr in listify(self.dependancy[nm])]
    #     A = self.get(path)
    #     if len(A)==0:
    #         Y = {k: self.X['all'][[]].join(y)['credit_hr'].unstack().dropna(how='all', axis=1).fillna(0) for k, y in self.reg_df['all'].items()}
    #         A = Y['cur'].rename(columns=lambda x:x+'_cur').join(Y['end']>0).prep()
    #     self.get(path, A)
    #     missing = [c for c in self.crse if c not in self.Y['all']]
    #     assert not missing, f'missing {missing}'
    #     return A

    # def get_mlt(self, path=None):
    #     nm = 'mlt'
    #     path = {'nm':nm, 'styp_code':'all'}
    #     [getattr(self,'get_'+attr)() for attr in listify(self.dependancy[nm])]
    #     A = self.get(path)
    #     if len(A)==0:
    #         mlt_grp = ['crse','levl_code','styp_code','pred_code']
    #         Y = {k: self.X['all'][[]].join(y)[['credit_hr']] for k, y in self.reg_df['all'].items()}
    #         agg = lambda y: y.groupby(mlt_grp)['credit_hr'].agg(lambda x: (x>0).sum())
    #         numer = agg(self.reg_df['all']['end'])
    #         denom = agg(Y['end'])
    #         M = (numer / denom).replace(np.inf, pd.NA).rename('mlt').reset_index().query(f"pred_code != {self.infer_term}").prep()
    #         M['mlt_code'] = M['pred_code']
    #         N = M.copy().assign(pred_code=self.infer_term)
    #         A = pd.concat([M, N], axis=0).set_index([*mlt_grp,'mlt_code'])
    #     self.get(path, A)
    #     return A
    
    
    # def get_inputs(self):
    #     path = ['inputs','all']
    #     A, b = self.get(path)
    #     repl = {'term_code':'pred_code', 'term_desc':'pred_desc'}

    #     opts = {x:self[x] for x in ['cycle_day','overwrite','show']}
    #     if 'term' not in A:
    #         A['term'] = {term_code: TERM(term_code=term_code, **opts).get_raw() for term_code in uniquify([*self.term_codes,self.infer_term])}

    #     if 'raw_df' not in A:
    #         print('getting raw_df')
    #         with warnings.catch_warnings(action='ignore'):
    #             A['raw_df'] = pd.concat([term.raw for term in A['term'].values()], ignore_index=True).dropna(axis=1, how='all').rename(columns=repl).prep()

    #     if 'reg_df' not in A:
    #         print('getting reg_df')
    #         with warnings.catch_warnings(action='ignore'):
    #             A['reg_df'] = {k: pd.concat([term.reg[k].query(f"crse in {self.crse}") for term in A['term'].values()]).rename(columns=repl).prep() for k in ['cur','end']}

    #     where = lambda x: x.query("levl_code == 'ug' and styp_code in ('n','r','t')").copy()
    #     if 'X' not in A:
    #         print('getting X')
    #         R = A['raw_df'].copy()
    #         repl = {'ae':0, 'n1':1, 'n2':2, 'n3':3, 'n4':4, 'r1':1, 'r2':2, 'r3':3, 'r4':4}
    #         R['hs_qrtl'] = pd.cut(R['hs_pctl'], bins=[-1,25,50,75,90,101], labels=[4,3,2,1,0], right=False).combine_first(R['apdc_code'].map(repl))
    #         R['remote'] = R['camp_code'] != 's'
    #         R['resd'] = R['resd_code'] == 'r'
    #         R['lgcy'] = ~R['lgcy_code'].isin(['n','o'])
    #         R['majr_code'] = R['majr_code'].replace({'0000':pd.NA, 'und':pd.NA, 'eled':'eted', 'agri':'unda'})
    #         R['coll_code'] = R['coll_code'].replace({'ae':'an', 'eh':'ed', 'hs':'hl', 'st':'sm', '00':pd.NA})
    #         R['coll_desc'] = R['coll_desc'].replace({
    #             'ag & environmental_sciences':'ag & natural_resources',
    #             'education & human development':'education',
    #             'health science & human_service':'health sciences',
    #             'science & technology':'science & mathematics'})
    #         majr = ['majr_desc','dept_code','dept_desc','coll_code','coll_desc']
    #         S = R.sort_values('cycle_date').drop_duplicates(subset='majr_code', keep='last')[['majr_code',*majr]]
    #         X = where(R.drop(columns=majr).merge(S, on='majr_code', how='left')).prep().prep_bool()

    #         checks = [
    #             'cycle_day >= 0',
    #             'apdc_day >= cycle_day',
    #             'appl_day >= apdc_day',
    #             'birth_day >= appl_day',
    #             'birth_day >= 5000',
    #             'distance >= 0',
    #             'hs_pctl >=0',
    #             'hs_pctl <= 100',
    #             'hs_qrtl >= 0',
    #             'hs_qrtl <= 4',
    #             'act_equiv >= 1',
    #             'act_equiv <= 36',
    #             'gap_score >= 0',
    #             'gap_score <= 100',
    #         ]
    #         for check in checks:
    #             mask = X.eval(check)
    #             assert mask.all(), [check,X[~mask].disp(5)]
    #         for k, v in self.fill.items():
    #             X[k] = X.impute(k, *listify(v))
    #         M = X.isnull().rename(columns=lambda x:x+'_missing')
    #         A['X'] = X.join(M).prep().prep_bool().set_index(self.attr, drop=False).rename(columns=lambda x:'__'+x)

    #     if 'Y' not in A:
    #         print('getting Y')
    #         mlt_grp = ['crse','levl_code','styp_code','pred_code']
    #         Y = {k: A['X'][[]].join(y.set_index(['pidm','pred_code','crse'])['credit_hr']) for k, y in A['reg_df'].items()}
    #         agg = lambda y: where(y).groupby(mlt_grp)['credit_hr'].agg(lambda x: (x>0).sum())
    #         numer = agg(A['reg_df']['end'])
    #         denom = agg(Y['end'])
    #         M = (numer / denom).replace(np.inf, pd.NA).rename('mlt').reset_index().query(f"pred_code != {self.infer_term}").prep()
    #         M['mlt_code'] = M['pred_code']
    #         N = M.copy().assign(pred_code=self.infer_term)
    #         A['mlt'] = pd.concat([M, N], axis=0).set_index([*mlt_grp,'mlt_code'])
    #         Y = {k: y.squeeze().unstack().dropna(how='all', axis=1).fillna(0) for k, y in Y.items()}
    #         A['Y'] = Y['cur'].rename(columns=lambda x:x+'_cur').join(Y['end']>0).prep()
        
    #     self.get(path, A)
    #     for k,v in A.items():
    #         self[k] = v
        
    #     missing = [c for c in self.crse if c not in self.Y]
    #     assert not missing, f'missing {missing}'





    # def get_transformed(self):
    #     grid = {'nm':'transformed', 'styp_code':self.styp_codes, 'term_code':'all', 'crse_code':'all'}
    #     [getattr(self,'get_'+attr)() for attr in listify(self.dependancy[grid['nm']])]
    #     start_time = time.perf_counter()
    #     I = cartesian(grid)
    #     for i, path in enumerate(I):
    #         A = self.get(path)
    #         if len(A)==0:

    #             J = self.trf_list
    #             print(len(J), end=": ")
    #             for j, par in enumerate(J):
    #                 p = stringify(par)
    #                 if p not in A:
    #                     trf = ColumnTransformer([(c,t,["__"+c]) for c,t in par.items()], remainder='drop', verbose_feature_names_out=False)
    #                     A[p] = trf.fit_transform(self.X.query(f"styp_code == @path['styp_code']")).prep().prep_bool().prep_category().sort_index()
    #                 print(j, end=", ")

    #         self.get(path, A)
    #         self.report(start_time, i, I)
        # return A


    # def get_transformed(self, path):
    #     path = path | {'nm':'transformed', 'crse':'all', 'train_code':'all'}
    #     A, b = self.get(path)
    #     if b:
    #         L = self.trf_list
    #         for k, par in enumerate(L):
    #             p = stringify(par)
    #             if p not in A:
    #                 trf = ColumnTransformer([(c,t,["__"+c]) for c,t in par.items()], remainder='drop', verbose_feature_names_out=False)
    #                 A[p] = trf.fit_transform(self.X.query(f"styp_code == @path['styp_code']")).prep().prep_bool().prep_category().sort_index()
    #             print(f"parameters {k+1} / {len(L)} = {(k+1)/len(L)*100:.2f}% complete")
    #         self.get(path, A)
    #     return A, b, path


    # def get_imputed(self, path):
    #     path = path | {'nm':'imputed', 'crse':'all', 'train_code':'all'}
    #     A, b = self.get(path)
    #     if b:
    #         L = self.imp_trf_list
    #         for k, par in enumerate(L):
    #             p = stringify(par)
    #             if p not in A:
    #                 q = stringify(par['trf'])
    #                 T = self.get_transformed(path)[0][q]
    #                 imp = self.get_model(T, par['imp'])
    #                 A[p] = pd.concat([imp.complete_data(k).addlevel('imp', k) for k in range(imp.dataset_count())])
    #             print(f"parameters {k+1} / {len(L)} = {(k+1)/len(L)*100:.2f}% complete")
    #         self.get(path, A)
    #     return A, b, path


    # def get_predicted(self, path):
    #     path = path | {'nm':'predicted'}
    #     A, b = self.get(path)
    #     if b:
    #         L = self.clf_imp_trf_list
    #         for k, par in enumerate(L):
    #             p = stringify(par)
    #             if p not in A:
    #                 q = stringify({k: par[k] for k in ['imp','trf']})
    #                 I = self.get_imputed(path)[0][q]
    #                 cols = uniquify(['_allcrse_cur', path['crse']+'_cur', path['crse']], False)
    #                 Z = I.join(self.Y[cols]).prep().prep_bool().prep_category().sort_index()
    #                 B = Z.copy()

    #                 actual = Z[path['crse']].copy().rename('actual').to_frame()
    #                 Z.loc[Z.eval(f"pred_code!=@path['train_code']"), path['crse']] = pd.NA
    #                 clf = self.get_model(Z, par['clf'])


    #                 Z.loc[:, path['crse']] = pd.NA
    #                 predicted = clf.impute_new_data(Z)
    #                 A[p] = pd.concat([actual
    #                             .assign(predicted=predicted.complete_data(k)[path['crse']])
    #                             .addlevel('crse', path['crse'])
    #                             .addlevel('train_code', path['train_code'])
    #                             .addlevel('sim', k)
    #                         for k in range(predicted.dataset_count())]).prep_bool()[['predicted','actual']]
    #                 return {'clf':clf, 'input':B, 'output':A[p]}
    #             print(f"parameters {k+1} / {len(L)} = {(k+1)/len(L)*100:.2f}% complete")
    #         self.get(path, A)
    #     return A, b, path

    # def get_transformed(self):
    #     grid = {'nm':'transformed', 'styp_code':self.styp_codes, 'train_code':'all', 'crse_code':'all'}
    #     def func(path, par):
    #         trf = ColumnTransformer([(c,t,["__"+c]) for c,t in par.items()], remainder='drop', verbose_feature_names_out=False)
    #         trf.output = trf.fit_transform(self['inputs']['X'].query(f"styp_code == @path['styp_code']")).prep().prep_bool().prep_category().sort_index()
    #         return trf
    #         # return trf.fit_transform(self.X['all'].query(f"styp_code == @path['styp_code']")).prep().prep_bool().prep_category().sort_index()
    #     self.run(grid, self.loop_par(self.trf_list, func))


    # def get_imputed(self):
    #     grid = {'nm':'imputed', 'styp_code':self.styp_codes, 'train_code':'all', 'crse_code':'all'}
    #     def func(path, par):
    #         trf = self.get(path | {'nm':'transformed', 'par':stringify(par['trf'])})
    #         imp = self.get_model(trf.output, par['imp'])
    #         imp.trf = trf
    #         imp.output = pd.concat([imp.complete_data(k).addlevel('imp', k) for k in range(imp.dataset_count())])
    #         return imp
    #     self.run(grid, self.loop_par(self.imp_trf_list, func))


    # def loop_par(self, par_list, inp_list, func):
    #     def g(path):
    #         A = dict()
    #         print(len(par_list)*len(inp_list), end=": ")
    #         for j, par, inp in enumerate(it.product(par_list, inp_list)):
    #             print(j, end=", ")
    #             a = func(path, par)
    #             a.params = {'idx': j} | par
    #             A[stringify(par)] = a
    #         print(j+1)
    #         return A
    #     return g


    
    # def get_transformed(self):
    #     grid = {'nm':'transformed', 'styp_code':self.styp_codes, 'train_code':'all', 'crse_code':'all'}
    #     def func(path):
    #         A = dict()
    #         for j, par in enumerate(self.trf_list):
    #         # for par in self.trf_list:
    #             trf = ColumnTransformer([(c,t,["__"+c]) for c,t in par.items()], remainder='drop', verbose_feature_names_out=False)
    #             trf.output = trf.fit_transform(self['inputs']['X'].query(f"styp_code == @path['styp_code']")).prep().prep_bool().prep_category().sort_index()
    #             trf.par = par
    #             trf.idx = j
    #             A[j] = trf
    #         return A
    #     self.run(grid, func)

            # return trf
            # return trf.fit_transform(self.X['all'].query(f"styp_code == @path['styp_code']")).prep().prep_bool().prep_category().sort_index()
        # self.run(grid, self.loop_par(self.trf_list, func))

    # def get_imputed(self):
    #     grid = {'nm':'imputed', 'styp_code':self.styp_codes, 'train_code':'all', 'crse_code':'all'}
    #     def func(path):
    #         A = dict()
    #         for j, par in enumerate(self.imp_list):
    #             A[j] = dict()
    #             for k, trf in enumerate(self.get(path | {'nm':'transformed'})):
    #                 imp = self.get_model(trf.output, par)
    #                 imp.output = pd.concat([imp.complete_data(k).addlevel('imp', k) for k in range(imp.dataset_count())])
    #                 imp.par = par
    #                 imp.idx = j
    #                 imp.trf = trf
    #                 A[j][k] = imp
    #         return A
    #     self.run(grid, func)
                    




        # for j, par in enumerate(self.trf_list):
        #     trf = ColumnTransformer([(c,t,["__"+c]) for c,t in par.items()], remainder='drop', verbose_feature_names_out=False)
        #     trf.output = trf.fit_transform(self['inputs']['X'].query(f"styp_code == @path['styp_code']")).prep().prep_bool().prep_category().sort_index()
        #     trf.par_idx = j
        #     trf.par = par


    # def get_imputed(self):
    #     grid = {'nm':'imputed', 'styp_code':self.styp_codes, 'train_code':'all', 'crse_code':'all'}
    #     def func(path, par, trf):

    #         trf = self.get(path | {'nm':'transformed', 'par':stringify(par['trf'])})
    #         imp = self.get_model(trf.transformed, par['imp'])
    #         imp.trf = trf
    #         imp.output = pd.concat([imp.complete_data(k).addlevel('imp', k) for k in range(imp.dataset_count())])
    #         return imp
    #     self.run(grid, self.loop_par(self.imp_trf_list, func))



    # def get_predicted(self):
    #     grid = {'nm':'predicted', 'styp_code':self.styp_codes, 'train_code':self.term_codes, 'crse_code':self.crse_codes}
    #     def func(path, par):
    #         imp = self.get(path | {'nm':'imputed', 'train_code':'all', 'crse_code':'all', 'par':stringify({k: par[k] for k in ['imp','trf']})})
    #         cols = uniquify(['_allcrse_cur', path['crse_code']+'_cur', path['crse_code']], False)
    #         Z = imp.output.join(self['inputs']['Y'][cols]).prep().prep_bool().prep_category().sort_index()
    #         actual = Z[path['crse_code']].copy().rename('actual').to_frame()
    #         Z.loc[Z.eval(f"pred_code!=@path['train_code']"), path['crse_code']] = pd.NA
    #         clf = self.get_model(Z, par['clf'])
    #         clf.imp = imp
    #         # clf.params = par
    #         # Z.loc[:, path['crse_code']] = pd.NA
    #         # P = clf.impute_new_data(Z)
    #         # clf.new_data = P
            
    #         clf.details = pd.concat([actual
    #                 .assign(predicted=clf.complete_data(k)[path['crse_code']],
    #                         proba=clf.get_raw_prediction(path['crse_code'], k))
    #                 .addlevel('crse_code', path['crse_code'])
    #                 .addlevel('train_code', path['train_code'])
    #                 .addlevel('sim', k)
    #             for k in range(clf.dataset_count())]).prep().prep_bool()[['proba','predicted','actual']]
    #         P = clf.details.join(self['inputs']['mlt']).reset_index().copy()
    #         clf.score = balanced_accuracy_score(P['actual'], P['predicted'])*100
    #         T = P[['pred_code','pred_desc']].drop_duplicates()
    #         for k in ['train','mlt']:
    #             P = P.merge(T.rename(columns=lambda x: x.replace('pred',k)))
    #         for k in ['predicted','actual']:
    #             P[k+'_mlt'] = P[k] * P['mlt']
    #         Q = P.copy().assign(styp_code='all', styp_desc='all incoming')
    #         clf.summary = (
    #             pd.concat([P,Q])
    #             .query('pred_code != train_code')
    #             .groupby(['crse_code','levl_code','levl_desc','styp_code','styp_desc','pred_code','pred_desc','train_code','train_desc','mlt_code','mlt_desc','imp','sim'])
    #             .apply(lambda x: pd.Series({
    #                 'predicted': x['predicted_mlt'].sum(),
    #                 'actual': x['actual_mlt'].sum(),
    #                 'error': x['predicted_mlt'].sum() - x['actual_mlt'].sum(),
    #                 'error_pct': (x['predicted_mlt'].sum() - x['actual_mlt'].sum()) / x['actual_mlt'].sum() * 100,
    #                 'acc_pct': accuracy_score(x['actual'], x['predicted'])*100,
    #                 'bal_acc_pct': balanced_accuracy_score(x['actual'], x['predicted'])*100,
    #                 'f1_pct': f1_score(x['actual'], x['predicted'], zero_division=np.nan)*100,
    #                 'mlt': x['mlt'].mean(),
    #                 # 'zero': x['mlt'].var(),
    #                 # 'mse_pct': ((1*x['predicted'] - x['actual'])**2).mean()*100,
    #                 # 'f1_inv_pct': (1-f1_score(x['actual'], x['predicted'], zero_division=np.nan))*100,
    #             }), include_groups=False)
    #         )
    #         return clf
    #     self.run(grid, self.loop_par(self.clf_imp_trf_list, func))


    # def get_optimal(self):
    #     grid = {'nm':'optimal', 'styp_code':self.styp_codes, 'train_code':self.term_codes, 'crse_code':self.crse_codes}
    #     def func(path):
    #         P = self.get(path | {'nm':'predicted'})
    #         D = {k:v.score for k,v in P.items()}
    #         par = min(D, key=D.get)
    #         return P[par]
    #     self.run(grid, func)


    # def loop_par(self, par_list, input_list, func):
    #     def g(path):
    #         # A = dict()
    #         A = []
    #         for j, par in enumerate(par_list):
    #             print(len(input_list), end=": ")
    #             for k, inp in enumerate(input_list):
    #                 print(k, end=", ")
    #                 a = func(path, par, inp)
    #                 a.idx = j
    #                 a.par = par



    #         L = list(it.product(*enumerate(par_list), input_list))
    #         print(len(L), end=": ")
    #         for j, par, inp in L:
    #             print(j, end=", ")
    #             a = func(path, par, inp)
    #             a.idx = j
    #             a.par = par
    #             # A[stringify(par)] = a
    #             A.append(a)
    #         print(j+1)
    #         return A
    #     return g



    # def loop_par(self, par_list, func):
    #     def g(path):
    #         A = dict()
    #         print(len(par_list), end=": ")
    #         for j, par in enumerate(par_list):
    #             print(j, end=", ")
    #             a = func(path, par)
    #             a.idx = j
    #             a.par = par
    #             A[stringify(par)] = a
    #         print(j+1)
    #         return A
    #     return g


    def __post_init__(self):
        self.root = root_path / f"resources/rslt/{rjust(self.cycle_day,3,0)}"
        mkdir(self.root)
        D = {'trm':False, 'adm':False, 'reg':False, 'flg':False, 'raw':False, 
             'terms':False, 'raw_df':False, 'reg_df':False, 'X':False, 'Y':False, 'mlt':False,
             'inputs':True,
             'transformed':False, 'imputed':False, 'predicted':False,
             'optimal':False,
             'details':False, 'summary':False, 'params':False,
             'outputs':False,
             }
        
        for x in ['overwrite','show']:
            self[x] = D.copy() if self[x] is None else D.copy() | self[x]
            

        self.dependancy = {
            'raw':['reg','adm','flg'],
            'terms':[],
            'raw_df':'terms',
            'reg_df':'terms',
            'X':'raw_df',
            'Y':['X','reg_df'],
            'mlt':'Y',
            'inputs':['terms','raw_df','reg_df','X','Y','mlt'],
            'transformed':'X',
            'imputed':'transformed',
            'predicted':['Y','mlt','imputed'],
            'optimal':'predicted',
            'details':'optimal',
            'summary':'optimal',
            'params':'optimal',
            'outputs':['details','summary','params'],
        }
        for dep, L in self.dependancy.items():
            for ind in listify(L):
                self.overwrite[dep] |= self.overwrite[ind]

        for k, v in self.overwrite.items():
            if v:
                delete(self.root / k)
        for k in ['fill','trf_grid','imp_grid']:
            if k not in self:
                self[k] = dict()

        self.crse_codes = uniquify(['_allcrse', *listify(self.crse_codes)])
        self.styp_codes = uniquify(self.styp_codes)
        self.term_codes = [x for x in uniquify(self.term_codes) if x != self.infer_term]

        self.trf_list = cartesian({k: uniquify(v, key=str) for k,v in self.trf_grid.items()})
        self.trf_list = [uniquify({k:v for k,v in t.items() if v not in ['drop',None,'']}) for t in self.trf_list]

        imp_default = {'datasets':5, 'iterations':3, 'tune':True, 'mmc':0, 'mmf':None}
        self.imp_list = cartesian(self.imp_grid)
        self.imp_list = [uniquify(imp_default|v) for v in self.imp_list]
        
        clf_default = {'datasets':5, 'iterations':3, 'tune':True, 'mmc':0, 'mmf':None}
        self.clf_list = cartesian(self.clf_grid)
        self.clf_list = [uniquify(clf_default | v) for v in self.clf_list]
        
        # self.imp_trf_list = mysort([uniquify({'imp':imp, 'trf':trf}) for imp, trf in it.product(self.imp_list, self.trf_list)], key=str)
        # self.clf_imp_trf_list = mysort([uniquify({'clf':clf, 'imp':imp, 'trf':trf}) for clf, imp, trf in it.product(self.clf_list, self.imp_list, self.trf_list)], key=str)


    def get_filename(self, path, suffix='.pkl'):
        return (self.root / join(path.values() if isinstance(path, dict) else path, '/')).with_suffix(suffix)

    def get(self, path, val=None, **kwargs):
        if val is not None:
            nest(path, self.__dict__, val)
            write(self.get_filename(path, **kwargs), val, overwrite=True)
        try:
            val = nest(path, self.__dict__)
        except:
            try:
                val = read(self.get_filename(path))
                nest(path, self.__dict__, val)
            except:
                val = None
        return val

    def run(self, grid, func):
        [getattr(self,'get_'+attr)() for attr in listify(self.dependancy[grid['nm']])]
        start_time = time.perf_counter()
        I = cartesian(grid, sort=False)
        for i, path in enumerate(I):
            A = self.get(path)
            if A is None:
                elapsed = (time.perf_counter() - start_time) / 60
                complete = i / len(I) if len(I) > 0 else 1
                rate = elapsed / i if i > 0 else 0
                remaining = rate * (len(I) - i)
                msg = f"{join(path.values())}; complete: {i} / {len(I)} = {complete*100:.2f}%; elapsed = {elapsed:.2f} min; remaining = {remaining:.2f} min @ {rate:.2f} min per model"
                print(msg)
                A = func(path.copy())
                self.get(path, A)
        return A
            

    def get_terms(self):
        grid = {'grp':'inputs', 'nm':'terms', 'term_code':uniquify([*self.term_codes,self.infer_term])}
        def func(path):
            opts = {x:self[x] for x in ['cycle_day','overwrite','show']}
            return TERM(term_code=path['term_code'], **opts).get_raw()
        self.run(grid, func)

    def where(self, df):
        return df.query("levl_code == 'ug' and styp_code in ('n','r','t')").copy().rename(columns={'term_code':'pred_code', 'term_desc':'pred_desc'})

    def get_raw_df(self):
        grid = {'grp':'inputs', 'nm':'raw_df'}
        def func(path):
            with warnings.catch_warnings(action='ignore'):
                return self.where(pd.concat([term.raw for term in self['inputs']['terms'].values()], ignore_index=True).dropna(axis=1, how='all')).prep()
        self.run(grid, func)

    def get_reg_df(self):
        grid = {'grp':'inputs', 'nm':'reg_df'}
        def func(path):
            with warnings.catch_warnings(action='ignore'):
                return {k: self.where(pd.concat([term.reg[k].query(f"crse_code in {self.crse_codes}") for term in self['inputs']['terms'].values()])).prep().set_index(['pidm','pred_code','crse_code']) for k in ['cur','end']}
        self.run(grid, func)

    def get_X(self):
        grid = {'grp':'inputs', 'nm':'X'}
        def func(path):
            R = self['inputs']['raw_df']
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
            return X.join(M).prep().prep_bool().set_index(self.attr, drop=False).rename(columns=lambda x:'__'+x)
        self.run(grid, func)

    def get_Y(self):
        grid = {'grp':'inputs', 'nm':'Y'}
        def func(path):
            Y = {k: self['inputs']['X'][[]].join(y)['credit_hr'].unstack().dropna(how='all', axis=1).fillna(0) for k, y in self['inputs']['reg_df'].items()}
            return Y['cur'].rename(columns=lambda x:x+'_cur').join(Y['end']>0).prep()
        self.run(grid, func)

    def get_mlt(self):
        grid = {'grp':'inputs', 'nm':'mlt'}
        def func(path):
            mlt_grp = ['crse_code','levl_code','styp_code','pred_code']
            Y = {k: self['inputs']['X'][[]].join(y)[['credit_hr']] for k, y in self['inputs']['reg_df'].items()}
            agg = lambda y: y.groupby(mlt_grp)['credit_hr'].agg(lambda x: (x>0).sum())
            numer = agg(self['inputs']['reg_df']['end'])
            denom = agg(Y['end'])
            M = (numer / denom).replace(np.inf, pd.NA).rename('mlt').reset_index().query(f"pred_code != {self.infer_term}").prep()
            M['mlt_code'] = M['pred_code']
            N = M.copy().assign(pred_code=self.infer_term)
            return pd.concat([M, N], axis=0).set_index([*mlt_grp,'mlt_code'])
        self.run(grid, func)


    def get_model(self, X, par, inspect=False):
        par = par.copy()
        iterations = par.pop('iterations')
        datasets = par.pop('datasets')
        tune = par.pop('tune')
        mmc = par.pop('mmc')
        mmf = par.pop('mmf')
        if mmc > 0 and mmf is not None:
            par['mean_match_scheme'] = getattr(mf, mmf).copy()
            par['mean_match_scheme'].set_mean_match_candidates(mmc)
        if tune:
            # print('tuning')
            model = mf.ImputationKernel(X, datasets=1, **par)
            model.mice(iterations=1)
            optimal_parameters, losses = model.tune_parameters(dataset=0, optimization_steps=5)
        else:
            # print('not tuning')
            optimal_parameters = None
        model = mf.ImputationKernel(X, datasets=datasets, **par)
        model.mice(iterations=iterations, variable_parameters=optimal_parameters)
        if inspect:
            model.inspect()
        return model


    def get_transformed(self):
        grid = {'nm':'transformed', 'styp_code':self.styp_codes, 'train_code':'all', 'crse_code':'all', 'trf_idx': range(len(self.trf_list))}
        def func(path):
            idx = path.pop('trf_idx')
            par = self.trf_list[idx]
            trf = ColumnTransformer([(c,t,["__"+c]) for c,t in par.items()], remainder='drop', verbose_feature_names_out=False)
            trf.idx = idx
            trf.par = par
            trf.output = trf.fit_transform(self['inputs']['X'].query(f"styp_code == @path['styp_code']")).prep().prep_bool().prep_category().sort_index()
            return trf
        self.run(grid, func)


    def get_imputed(self):
        grid = {'nm':'imputed', 'styp_code':self.styp_codes, 'train_code':'all', 'crse_code':'all', 'trf_idx': range(len(self.trf_list)), 'imp_idx': range(len(self.imp_list))}
        def func(path):
            idx = path.pop('imp_idx')
            par = self.imp_list[idx]
            trf = self.get(path | {'nm':'transformed'})
            imp = self.get_model(trf.output, par)
            imp.idx = idx
            imp.par = par
            imp.trf = trf
            imp.output = pd.concat([imp.complete_data(k).addlevel('imp', k) for k in range(imp.dataset_count())])
            return imp
        self.run(grid, func)


    def get_predicted(self):
        grid = {'nm':'predicted', 'styp_code':self.styp_codes, 'train_code':self.term_codes, 'crse_code':self.crse_codes, 'trf_idx': range(len(self.trf_list)), 'imp_idx': range(len(self.imp_list)), 'clf_idx': range(len(self.clf_list))}
        def func(path):
            idx = path.pop('clf_idx')
            par = self.clf_list[idx]
            imp = self.get(path | {'nm':'imputed', 'train_code':'all', 'crse_code':'all'})
            cols = uniquify(['_allcrse_cur', path['crse_code']+'_cur', path['crse_code']], False)
            Z = imp.output.join(self['inputs']['Y'][cols]).prep().prep_bool().prep_category().sort_index()
            actual = Z[path['crse_code']].copy().rename('actual').to_frame()
            Z.loc[Z.eval(f"pred_code!=@path['train_code']"), path['crse_code']] = pd.NA
            clf = self.get_model(Z, par)
            clf.idx = idx
            clf.par = par
            clf.imp = imp

            clf.details = pd.concat([actual
                    .assign(predicted=clf.complete_data(k)[path['crse_code']],
                            proba=clf.get_raw_prediction(path['crse_code'], k))
                    .addlevel('crse_code', path['crse_code'])
                    .addlevel('train_code', path['train_code'])
                    .addlevel('sim', k)
                for k in range(clf.dataset_count())]).prep().prep_bool()[['proba','predicted','actual']]
            P = clf.details.join(self['inputs']['mlt']).reset_index().copy()
            clf.score = balanced_accuracy_score(P['actual'], P['predicted'])*100
            T = P[['pred_code','pred_desc']].drop_duplicates()
            for k in ['train','mlt']:
                P = P.merge(T.rename(columns=lambda x: x.replace('pred',k)))
            for k in ['predicted','actual']:
                P[k+'_mlt'] = P[k] * P['mlt']
            Q = P.copy().assign(styp_code='all', styp_desc='all incoming')
            clf.summary = (
                pd.concat([P,Q])
                .query('pred_code != train_code')
                .groupby(['crse_code','levl_code','levl_desc','styp_code','styp_desc','pred_code','pred_desc','train_code','train_desc','mlt_code','mlt_desc','imp','sim'])
                .apply(lambda x: pd.Series({
                    'predicted': x['predicted_mlt'].sum(),
                    'actual': x['actual_mlt'].sum(),
                    'error': x['predicted_mlt'].sum() - x['actual_mlt'].sum(),
                    'error_pct': (x['predicted_mlt'].sum() - x['actual_mlt'].sum()) / x['actual_mlt'].sum() * 100,
                    'acc_pct': accuracy_score(x['actual'], x['predicted'])*100,
                    'bal_acc_pct': balanced_accuracy_score(x['actual'], x['predicted'])*100,
                    'f1_pct': f1_score(x['actual'], x['predicted'], zero_division=np.nan)*100,
                    'mlt': x['mlt'].mean(),
                    # 'zero': x['mlt'].var(),
                    # 'mse_pct': ((1*x['predicted'] - x['actual'])**2).mean()*100,
                    # 'f1_inv_pct': (1-f1_score(x['actual'], x['predicted'], zero_division=np.nan))*100,
                }), include_groups=False)
            )
            return clf
        self.run(grid, func)


    def get_optimal(self):
        grid = {'nm':'optimal', 'styp_code':self.styp_codes, 'train_code':self.term_codes, 'crse_code':self.crse_codes}
        def func(path):
            D = self.get(path | {'nm':'predicted'})
            E = [A for C in D.values() for B in C.values() for A in B.values()]
            return max(E, key=lambda e: e.score)
        self.run(grid, func)


    def get_outputs(self):
        for nm in ['details', 'summary', 'params']:
            grid = {'grp':'outputs', 'nm':nm}
            if nm != 'params':
                def func(path):
                    return pd.concat([getattr(C, nm) for S in self.optimal.values() for T in S.values() for C in T.values()])
            else:
                def func(path):
                    return pd.DataFrame([{'crse_code':crse_code, 'styp_code':styp_code, 'train_code':train_code, 'trf_idx': trf_idx, 'imp_idx': imp_idx, 'clf_idx': clf_idx,
                        **{f'trf_{key}': stringify(val) for key, val in clf.imp.trf.par.items()},
                        **{f'imp_{key}': stringify(val) for key, val in clf.imp.par.items()},
                        **{f'clf_{key}': stringify(val) for key, val in clf.par.items()},
                        'score': clf.score,
                        } for styp_code, S in self.predicted.items() for train_code, T in S.items() for crse_code, C in T.items() for trf_idx, trf in C.items() for imp_idx, imp in trf.items() for clf_idx, clf in imp.items()])
            A = self.run(grid, func)
            if nm != 'details':
                write(self.get_filename(grid, suffix='.csv'), A)


    def push(self):
        target_url = 'https://prod-121.westus.logic.azure.com:443/workflows/784fef9d36024a6abf605d1376865784/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=1Yrr4tE1SwYZ88SU9_ixG-WEdN1GFicqJwH_KiCZ70M'
        grid = {'grp':'outputs', 'nm':'summary'}
        with open(self.get_filename(grid, suffix='.csv'), 'rb') as target_file:
            response = requests.post(target_url, files = {"amp_summary.csv": target_file})
        print('file pushed')


code_desc = lambda x: [x+'_code', x+'_desc']
passthru = ['passthrough']
passdrop = ['passthrough', 'drop']
# passdrop = passthru
bintrf = lambda n_bins: KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None)
pwrtrf = make_pipeline(StandardScaler(), PowerTransformer())

kwargs = {
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
    # 'cycle_day': 168,
    'crse_codes': [
        # 'agec2317',
        # 'agri1100',
        # 'agri1419',
        # 'ansc1319',
        'arts1301',
        'biol1406',
        # # 'biol2401',
        # 'busi1301',
        'comm1311',
        # # 'comm1315',
        'engl1301',
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
        'birth_day': [*passthru, pwrtrf],
        # 'camp_code': passdrop,
        'coll_code': passdrop,
        'distance': [*passthru, pwrtrf],
        # 'fafsa_app': passthru,
        # 'finaid_accepted': passthru,
        'gap_score': passthru,
        'gender': passthru,
        'hs_qrtl': passthru,
        'international': passthru,
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
        # 'datasets': 2, 'iterations': 1, 'tune': False,
    },
    'clf_grid': {
        # 'datasets': 2, 'iterations': 1, 'tune': False,
    },

    'overwrite': {
        # 'trm':True,
        # 'reg':True,
        # 'adm':True,
        # 'flg':True,
        # 'raw':True,
        # 'terms': True,
        # 'raw_df': True,
        # 'reg_df': True,
        # 'X': True,
        # 'Y': True,
        # 'mlt': True,
        ## 'inputs': True,
        # 'transformed': True,
        # 'imputed': True,
        # 'predicted': True,
        # 'optimal': True,
        # 'details': True,
        # 'summary': True,
        # 'params': True,
        ## 'outputs': True,
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
        self.get_outputs()
        self.push()