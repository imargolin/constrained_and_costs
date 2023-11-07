import numpy as np
import pandas as pd
from sklearn import metrics
import timeit
from sklearn.metrics import confusion_matrix
from sklearn.utils import column_or_1d
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold

def cost_loss(y_true, y_pred, cost_mat):

    y_true = column_or_1d(y_true)
    y_true = (y_true == 1).astype(np.float32)
    y_pred = column_or_1d(y_pred)
    y_pred = (y_pred == 1).astype(np.float32)
    cost = y_true * ((1 - y_pred) * cost_mat[:, 1] + y_pred * cost_mat[:, 2])
    cost += (1 - y_true) * (y_pred * cost_mat[:, 0] + (1 - y_pred) * cost_mat[:, 3])
    return np.sum(cost)


def cost_mat(fp,fn,y):
    cost_mat = np.zeros((len(y),4))
    #false positives cost
    cost_mat[:,0]=fp
    #false negatives cost
    cost_mat[:,1]=fn
    return cost_mat

# prediction up to the constraint - without threshold
def prediction_up_to_constraint(y_pred_probs,constraint):
    D={}
    for val in y_pred_probs:
        D[val] = D.get(val,0) +1
    dictionary_items = D.items()
    sorted_D = sorted(dictionary_items,reverse=True)
    sorted_k = [sorted_D[i][0] for i in range(len(sorted_D))]
    np.random.seed(23)
    number_of_positive = 0
    y_pred = np.zeros(len(y_pred_probs))
    classified = []
    for s_i in sorted_k:
        for i,y in enumerate(y_pred_probs):
            if number_of_positive >= constraint:
                return y_pred
            if i in classified:
                continue
            if y >= s_i:
                y_pred[i] = 1
                classified.append(i)
                number_of_positive +=1
    return y_pred

# get the dynamic threshold
def get_dynamic_threshold(y_pred_probs,constraint, t):
    D={}
    for val in y_pred_probs:
        D[val] = D.get(val,0) +1
    dictionary_items = D.items()
    sorted_D = sorted(dictionary_items,reverse=True)
    sorted_k = [sorted_D[i][0] for i in range(len(sorted_D))]
    np.random.seed(23)
    number_of_positive = 0
    y_pred = np.zeros(len(y_pred_probs))
    classified = []
    dynamic_t = t
    for s_i in sorted_k: #list of thresholds, from higher to lower
        if s_i < t:
            return dynamic_t
        for i,y in enumerate(y_pred_probs):
            if number_of_positive >= constraint:
                dynamic_t = s_i
                return dynamic_t
            if i in classified:
                continue
            if y >= s_i:
                y_pred[i] = 1
                classified.append(i)
                number_of_positive +=1
    return dynamic_t

# cfp change in each iteration and a new model is built
def cost_update_iterations(cfp, cfn, clf, X_train, y_train, X_test, y_test, constraint, cost_mat_test, n_iteration=30):
    models_pred_probs, models_pred_probs_test = {}, {}
    times_dict, optimal_dict, const_dict_train, const_dict_test = {}, {}, {}, {}
    cost_mat_train = cost_mat(cfp,cfn,y_train)
    for c in constraint:
        print('c',c)
        start = timeit.default_timer()
        cfp_i = 0
        itr_dict, itr_dict_test = {}, {}
        for i in range(n_iteration): # need to check if the number of iteration here is enough to get to the stopping condition
            cfp_i +=1
            #print(cfp_i)
            cost_mat_train_i = cost_mat(cfp_i,cfn,y_train)
            y_pred_probs_i = models_pred_probs.get(i,0)
            y_pred_probs_i_test = models_pred_probs_test.get(i,0)
            t_i = cfp_i/(cfp_i+cfn)
            if y_pred_probs_i == 0:
                s = timeit.default_timer()
                clf.fit(np.array(X_train),np.array(y_train), cost_mat_train_i)
                #y_pred_i = clf.predict(np.array(X_train))
                y_pred_proba_i= clf.predict_proba(np.array(X_train))
                y_pred_probs_i = [y_pred_proba_i[j][1] for j in range(len(y_pred_proba_i))]
                models_pred_probs[i] = y_pred_probs_i
                s2 = timeit.default_timer()
                times_dict[i] = s2-s
                #y_pred_i_test = clf.predict(np.array(X_test))
                y_pred_proba_i_test= clf.predict_proba(np.array(X_test))
                y_pred_probs_i_test = [y_pred_proba_i_test[j][1] for j in range(len(y_pred_proba_i_test))]
                models_pred_probs_test[i] = y_pred_probs_i_test

            d_t_i = get_dynamic_threshold(y_pred_probs_i,c*2,t_i)
            y_pred_c_i = prediction_up_to_constraint(y_pred_probs_i,c*2)
            y_pred_c_i_test = prediction_up_to_constraint(y_pred_probs_i_test,c)
            if d_t_i <= t_i:
                have_optimal = optimal_dict.get(c,0)
                if have_optimal == 0:
                    y_pred_proba_i_optimal= clf.predict_proba(np.array(X_test))
                    y_pred_probs_i_optimal = [y_pred_proba_i[j][1] for j in range(len(y_pred_proba_i_optimal))]
                    y_pred_c_i_optimal = prediction_up_to_constraint(y_pred_probs_i_optimal,c)
                    cm_i = pd.DataFrame(confusion_matrix(y_test,y_pred_c_i_optimal))
                    cost_i = cost_loss(y_test,y_pred_c_i_optimal,cost_mat_test)
                    acc_i = metrics.accuracy_score(y_test, y_pred_c_i_optimal)
                    prc_i = metrics.precision_score(y_test, y_pred_c_i_optimal)
                    f_score_i = f1_score(y_test, y_pred_c_i_optimal)
                    stop = timeit.default_timer()
                    time = stop - start
                    optimal_dict[c] = {'cost': cost_i, 'itr': i, 'cm':cm_i, 'accuracy': acc_i, 'precision':prc_i,
                                       'f_score': f_score_i, 'time': time, 'd_t':d_t_i, 'probs':y_pred_probs_i}
                    itr_dict[i] = {'cost': cost_i,'cm':cm_i, 'accuracy': acc_i, 'precision':prc_i,
                                       'f_score': f_score_i, 'd_t':d_t_i, 'probs':y_pred_probs_i}
                    print('break')

                # break here to use the stopping condition
                #break

            cm_i = pd.DataFrame(confusion_matrix(y_train,y_pred_c_i))
            cost_i = cost_loss(y_train,y_pred_c_i,cost_mat_train)
            acc_i = metrics.accuracy_score(y_train, y_pred_c_i)
            prc_i = metrics.precision_score(y_train, y_pred_c_i)
            f_score_i = f1_score(y_train, y_pred_c_i)
            itr_dict[i] = {'cost': cost_i,'cm':cm_i, 'accuracy': acc_i, 'precision':prc_i,
                                       'f_score': f_score_i, 'd_t':d_t_i, 'probs':y_pred_probs_i}

            cm_i_test = pd.DataFrame(confusion_matrix(y_test,y_pred_c_i_test))
            cost_i_test = cost_loss(y_test,y_pred_c_i_test,cost_mat_test)
            acc_i_test = metrics.accuracy_score(y_test, y_pred_c_i_test)
            prc_i_test = metrics.precision_score(y_test, y_pred_c_i_test)
            f_score_i_test = f1_score(y_test, y_pred_c_i_test)
            itr_dict_test[i] = {'cost': cost_i_test,'cm':cm_i_test, 'accuracy': acc_i_test, 'precision':prc_i_test,
                                  'f_score': f_score_i_test, 'd_t':d_t_i, 'probs':y_pred_probs_i_test}
        stop_c = timeit.default_timer()
        time_c = stop_c - start
        const_dict_train[c] = {'itr_dict': itr_dict, 'time': times_dict}
        const_dict_test[c] = {'itr_dict': itr_dict_test}


    return optimal_dict, const_dict_train, const_dict_test

# all the experimnt steps - DT, CS-DT, AdaSCL-WRC
def experiment(clf_cs_dt, clf_dt, random_state, X, y, cfp, cfn, constraint):
    kf = StratifiedKFold(n_splits=3, random_state=random_state, shuffle=True)
    fold_dict ={}
    fold = 1
    for train_index, test_index in kf.split(X,y):
        print(fold)
        cs_dict, cfp_dict, dt_dict  = {}, {}, {}
        cfp_dict = {}
        dt_dict = {}
        #fit models
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        cost_mat_train = cost_mat(cfp,cfn,y_train)
        cost_mat_test = cost_mat(cfp,cfn,y_test)
        start1 = timeit.default_timer()
        start_cs = timeit.default_timer()
        clf_cs_dt.fit(np.array(X_train),np.array(y_train), cost_mat_train)
        y_pred = clf_cs_dt.predict(np.array(X_test))
        y_pred_proba = clf_cs_dt.predict_proba(np.array(X_test))
        y_pred_probs = [y_pred_proba[j][1] for j in range(len(y_pred_proba))]
        stop_cs = timeit.default_timer()
        time_cs = stop_cs - start_cs
        start_dt = timeit.default_timer()
        clf_dt.fit(X_train,y_train)
        y_pred_dt = clf_dt.predict(X_test)
        y_pred_proba_dt = clf_dt.predict_proba(X_test)
        y_pred_probs_dt = [y_pred_proba_dt[k][1] for k in range(len(y_pred_proba_dt))]
        stop_dt = timeit.default_timer()
        time_dt = stop_dt - start_dt
        # add metrics without constraint
        #cost-sensitive model
        cm = pd.DataFrame(confusion_matrix(y_test,y_pred))
        n_positive = sum(y_pred)
        cost = cost_loss(y_test,y_pred,cost_mat_test)
        acc = metrics.accuracy_score(y_test, y_pred)
        prc = metrics.precision_score(y_test, y_pred)
        f_score = f1_score(y_test, y_pred)
        cs_dict[0] = {'cost': cost,'cm':cm, 'accuracy': acc,'precision':prc,
                      'f_score': f_score, 'time': time_cs,  'n_positive': n_positive}
        #constraint_dict[0] = {'cost': cost,'cm':cm, 'accuracy': acc, 'd_t':d_t}
        #decision tree model
        cm_dt = pd.DataFrame(confusion_matrix(y_test,y_pred_dt))
        model_number_of_positive = sum(y_pred_dt)
        cost_dt = cost_loss(y_test,y_pred_dt,cost_mat_test)
        acc_dt = metrics.accuracy_score(y_test, y_pred_dt)
        prc_dt = metrics.precision_score(y_test, y_pred_dt)
        f_score_dt = f1_score(y_test, y_pred_dt)
        dt_dict[0] = {'cost': cost_dt,'cm':cm_dt, 'accuracy': acc_dt,'precision':prc_dt,
                                'f_score': f_score_dt, 'time': time_dt, 'n_positive': model_number_of_positive }

        #add metrics with constraint
        for c in constraint:
            #cs
            start_cs_2 = timeit.default_timer()
            y_pred_c = prediction_up_to_constraint(y_pred_probs,c)
            stop_cs_2 = timeit.default_timer()
            time_cs_2 = stop_cs_2 - start_cs_2 + time_cs
            cm_c = pd.DataFrame(confusion_matrix(y_test,y_pred_c))
            cost = cost_loss(y_test,y_pred_c,cost_mat_test)
            acc = metrics.accuracy_score(y_test, y_pred_c)
            prc = metrics.precision_score(y_test, y_pred_c)
            f_score = f1_score(y_test, y_pred_c)
            cs_dict[c] = {'cost': cost,'cm':cm_c, 'accuracy': acc, 'precision':prc,
                          'f_score': f_score, 'time': time_cs_2, 'probs':y_pred_probs}
            #dt
            start_dt_2 = timeit.default_timer()
            y_pred_c_dt = prediction_up_to_constraint(y_pred_probs_dt, c)
            stop_dt_2 = timeit.default_timer()
            time_dt_2 = stop_dt_2 - start_dt_2 + time_dt
            cm_c_dt = pd.DataFrame(confusion_matrix(y_test, y_pred_c_dt))
            cost_c_dt = cost_loss(y_test, y_pred_c_dt, cost_mat_test)
            acc_c_dt = metrics.accuracy_score(y_test, y_pred_c_dt)
            prc_c_dt = metrics.precision_score(y_test, y_pred_c_dt)
            f_score_c_dt = f1_score(y_test, y_pred_c_dt)
            dt_dict[c] = {'cost': cost_c_dt, 'cm': cm_c_dt, 'accuracy': acc_c_dt, 'precision': prc_c_dt,
                         'f_score': f_score_c_dt, 'time': time_dt_2, 'probs': y_pred_probs_dt}


        optimal_dict, const_dict_train, const_dict_test = cost_update_iterations(cfp, cfn, clf_cs_dt, X_train, y_train,
                                                                                 X_test, y_test, constraint, cost_mat_test)

        fold_dict[fold] = {'cs': cs_dict, 'dt': dt_dict, 'optimal': optimal_dict,
                          'const_dict_train':const_dict_train,'const_dict_test': const_dict_test}
        fold += 1
    return fold_dict

def get_mean_list(list_):
    arrays = [np.array(x) for x in list_]
    return [np.mean(k) for k in zip(*arrays)]

def get_measure_list(data_dict, measure_name, random_state_list, constraint):
    cs_measure, dt_measure, one_before_measure, itr_list = [], [], [], []
    for random_state in random_state_list:
        for k in range(1,4):
            cs_measure_list, dt_measure_list, one_before_measure_list, itr_ = [], [], [], []
            for c in constraint:
                cs_measure_list.append(data_dict[random_state][k]['cs'][c][measure_name])
                dt_measure_list.append(data_dict[random_state][k]['dt'][c][measure_name])

                if data_dict[random_state][k]['optimal'].get(c,0) == 0:
                    itr = 29 # n_iteratin -1
                else:
                    itr = data_dict[random_state][k]['optimal'][c]['itr']
                itr_.append(itr)
                if itr == 0:
                    if measure_name == 'time':
                        one_before_measure_list.append(data_dict[random_state][k]['const_dict_train'][c][measure_name][0])
                    else:
                        one_before_measure_list.append(data_dict[random_state][k]['const_dict_test'][c]['itr_dict'][itr][measure_name])
                else:
                    if measure_name == 'time':
                        total_time = 0
                        for i in range(itr):
                            total_time += data_dict[random_state][k]['const_dict_train'][c][measure_name][i]
                        one_before_measure_list.append(total_time)
                    else:
                        one_before_measure_list.append(data_dict[random_state][k]['const_dict_test'][c]['itr_dict'][itr-1][measure_name])
            cs_measure.append(cs_measure_list)
            dt_measure.append(dt_measure_list)
            one_before_measure.append(one_before_measure_list)
            itr_list.append(itr_)
    return cs_measure, dt_measure, one_before_measure, itr_list

def plot_results(cs_list, dt_list, one_before_list, measure_name, idx, min_, max_, constraint, db_size):
    f = plt.figure()
    print(db_size)
    cs_mean = get_mean_list(cs_list)
    dt_mean = get_mean_list(dt_list)
    one_before_mean = get_mean_list(one_before_list)
    plt.plot(constraint[:idx+1], one_before_mean[:idx+1], label = 'AdaCSL-WRC')
    plt.plot(constraint[:idx+1], cs_mean[:idx+1] ,'--', c='purple', label = 'CS-DT')
    plt.plot(constraint[:idx+1], dt_mean[:idx+1] ,'-.', c='darkblue', label = 'DT')
    plt.xlabel('Constraint in %', fontsize=16)
    plt.ylabel(measure_name, fontsize=16)
    plt.ylim(min(min_)*0.9, max(max_)*1.05)
    #plt.xticks(constraint_range,constraint_in_precent)
    plt.legend(fontsize=12)
    plt.show()
#     fig_name = db_size + '_' + measure_name +'.png'
#     f.savefig(fig_name, bbox_inches='tight', dpi=600)