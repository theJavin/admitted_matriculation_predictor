from term import *
import requests
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


    def dump(self):
        return write(self.rslt, self, overwrite=True)

    def __post_init__(self):
        self.path = root_path / f"resources/rslt/{rjust(self.cycle_day,3,0)}"
        mkdir(self.path)
        self.rslt = self.path / f"rslt.pkl"
        self.summary = self.path / 'amp_summary.csv'
        self.details = self.path / 'amp_details.csv'
        # self.rslt = root_path / f"resources/rslt/{rjust(self.cycle_day,3,0)}

        
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
        self.crse = sorted({'_total', *listify(self.crse)})
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
            M['mlt_term'] = M['term_code']
            N = M.copy().assign(term_code=self.infer_term)
            self.mlt = pd.concat([M, N], axis=0).set_index([*self.mlt_grp,'mlt_term'])
            Y = {k: y.squeeze().unstack().dropna(how='all', axis=1).fillna(0) for k, y in Y.items()}
            self.Y = Y['cur'].rename(columns=lambda x:x+'_cur').join(Y['end']>0).prep()
        self.dump()


    def predict(self, crse, styp_code, params, train_term):
        print(ljust(crse,8), styp_code, train_term, 'creating')
        X = self.X.copy()
        if styp_code != 'all':
            X = X.query(f"styp_code=='{styp_code}'")
        trf = ColumnTransformer([(c,t,["__"+c]) for c,t in params['trf'].items()], remainder='drop', verbose_feature_names_out=False)
        cols = uniquify(['_total_cur',crse+'_cur',crse])
        Z = trf.fit_transform(X).join(self.Y[cols]).prep().prep_bool().prep_category().sort_index()
        y = Z[crse].copy().rename('actual').to_frame()
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
                .assign(predict=P.complete_data(k)[crse], train_term=train_term, crse=crse, sim=k)
                .set_index(['train_term','crse','sim'], append=True)
            for k in range(P.dataset_count())]).prep_bool()
        return details
    
    def aggregate(self, details):
        agg = lambda x: pd.Series({
            'predict': x['predict'].sum(min_count=1),
            'actual': x['actual'].sum(min_count=1),
            'mse_pct': ((1*x['predict'] - x['actual'])**2).mean()*100,
            'f1_inv_pct': (1-f1_score(x.dropna()['actual'], x.dropna()['predict'], zero_division=np.nan))*100,
        })
        summary = details.groupby([*self.mlt_grp,'train_term','sim']).apply(agg).join(self.mlt)#.rename_axis(index={'term_code':'pred_term'})
        for x in ['predict','actual']:
            summary[x] = summary[x] * summary['mlt']
        summary.insert(2, 'error', summary['predict'] - summary['actual'])
        summary.insert(3, 'error_pct', summary['error'] / summary['actual'] * 100)
        return summary
        # S = {'details':details, 'summary':summary.drop(columns='mlt').prep()}#, 'trf':trf, 'imp':imp}
        # S['summary'].disp(5)
        # return S
        # return S, True


    def analyze(self, df):
        def pivot(df, val):
            Y = (
                df
                .query(f"term_code!=train_term")
                .reset_index()
                .pivot_table(columns='train_term', index=['crse','styp_code','term_code'], values=val, aggfunc=['count',pctl(0),pctl(25),pctl(50),pctl(75),pctl(100)])
                .rename_axis(columns=[val,'train_term'])
                .stack(0, future_stack=True)
                .assign(abs_mean = lambda x: x.abs().mean(axis=1))
            )
            return Y
        mask = df.eval(f"term_code!={self.infer_term}")
        return {stat: pivot(df[mask], stat) for stat in ["predict","error","error_pct","mse_pct","f1_inv_pct"]} | {"project": pivot(df[~mask], "predict")}


    def main(self, styp_codes=('n','t','r')):
        self.preprocess()
        styp_codes = listify(styp_codes)
        g = lambda Y: Y | {k: pd.concat([y[k] for y in Y.values() if isinstance(y, dict) and k in y.keys()]).sort_index() for k in ['details','summary']}
        L = len(self.crse) * len(styp_codes) * len(self.params_list)
        k = 0
        start_time = time.perf_counter()
        self.optimal = dict()
        for crse in self.crse:
            for styp_code in listify(styp_codes):
                for params_idx, params in enumerate(self.params_list):
                    print("\n========================================================================================================\n")
                    print(ljust(crse,8),styp_code,params_idx)
                    new = False
                    for train_term in self.term_codes:
                        path = [crse,styp_code,str(params),train_term,'details']
                        try:
                            details = nest(path, self.pred)
                        except:
                            new = True
                            nest(path[:-1], self.pred, dict())
                            details = self.predict(crse, styp_code, copy.deepcopy(params), train_term)
                            nest(path, self.pred, details)
                        path[-1] = 'summary'
                        try:
                            summary = nest(path, self.pred)
                        except:
                            summary = self.aggregate(details)
                            nest(path, self.pred, summary)
                            self.dump()
                    Y = nest(path[:-2], self.pred)
                    for key in ['details', 'summary']:
                        Y[key] = pd.concat([y[key] for y in Y.values() if isinstance(y, dict) and key in y.keys()]).sort_index()
                    Y['rslt'] = self.analyze(Y['summary'])
                    if new:
                        # self.dump()
                        k += 1
                    else:
                        L -= 1
                    # Y['rslt']['error_pct'].query("error_pct == ' 50%'").round(decimals=2).disp(100)
                    E = Y['summary'].query(f"term_code!=train_term & term_code!={self.infer_term}")["error_pct"].abs()
                    # E.describe().to_frame().T.round(decimals=2).disp(200)
                    new = Y | {'params_idx':params_idx, 'params':params, 'score':E.median()}
                    print(f"new score = {new['score']:.2f}")
                    if pd.notnull(new['score']) and new['score'] < 30:
                        try:
                            old = nest(path[:-3], self.optimal)
                            print(f"old score = {old['score']:.2f}")
                            if new['score'] < old['score']:
                                print('replacing')
                                nest(path[:-3], self.optimal, new)
                            else:
                                print('keeping')
                        except:
                            nest(path[:-3], self.optimal, new)
                    elapsed = (time.perf_counter() - start_time) / 60
                    complete = k / L if L > 0 else 1
                    rate = elapsed / k if k > 0 else 0
                    remaining = rate * (L - k)
                    print(f"{k} / {L} = {complete*100:.2f}% complete, elapsed = {elapsed:.2f} min, remaining = {remaining:.2f} min @ {rate:.2f} min per model")
            self.dump()
        self.push()

    def combine(self):
        # for key in ['details', 'summary']:
        key = 'summary'
        A = pd.concat([S[key] for crse, C in self.optimal.items() for styp_code, S in C.items() if isinstance(S, dict) and key in S.keys()])
        if key == 'summary':
            B = A.copy().reset_index().assign(styp_code=A.reset_index()['styp_code'].replace({'n':'new first time','t':'transfer','r':'returning'}))
            C = B.assign(styp_code='all').groupby(A.index.names)[['predict','actual','error']].sum().reset_index()
            C['error_pct'] = C['error'] / C['actual'] * 100
            A = pd.concat([B,C])
        self.optimal[key] = A
        write(self[key], self.optimal[key], index=False)
        self.dump()

    def push(self):
        self.combine()
        target_url = 'https://prod-121.westus.logic.azure.com:443/workflows/784fef9d36024a6abf605d1376865784/triggers/manual/paths/invoke?api-version=2016-06-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=1Yrr4tE1SwYZ88SU9_ixG-WEdN1GFicqJwH_KiCZ70M'
        with open(self.summary, 'rb') as target_file:
            response = requests.post(target_url, files = {"amp_summary.csv": target_file})
        print('file pushed')

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
    'cycle_day': 179,
    'crse': [
        'agec2317',
        'agri1100',
        'agri1419',
        'ansc1319',
        'arts1301',
        'biol1406',
        'biol2401',
        'busi1301',
        'comm1311',
        'comm1315',
        'engl1301',
        'govt2305',
        'govt2306',
        'hist1301',
        'math1314',
        'math1324',
        'math1342',
        'math2412',
        'math2413',
        'psyc2301',
        'univ0204',
        'univ0301',
        'univ0304',
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
        'datasets': 10,
        # 'datasets': 1,
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
        # 'pred': True,
    },
    # 'inspect': True,
}


if __name__ == "__main__":
    @pd_ext
    def disp(df, max_rows=4, max_cols=200, **kwargs):
        display(HTML(df.to_html(max_rows=max_rows, max_cols=max_cols, **kwargs)))
        print(df.head(max_rows).reset_index().to_markdown(tablefmt='psql'))

    from IPython.utils.io import Tee
    self = AMP(**kwargs)
    with contextlib.closing(Tee(self.rslt.with_suffix('.txt'), "w", channel="stdout")) as outputstream:
        print(pd.Timestamp.now())
        self.preprocess()
        self.main()
        self.push()