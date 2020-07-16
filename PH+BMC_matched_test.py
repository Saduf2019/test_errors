
import os
import tensorflow as tf
import numpy as np
import scipy.io
from datetime import datetime
import pandas as pd
from lifelines.utils import concordance_index

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

sequence = 'data1'
momentum = 0.95
learning_rate_decay = 0.95
learning_rate_base = 1e-3
learning_rate_step = 80
warmup_step = 80

alpha = 1.5
belta = 1.0
gamma = 1.0
sita = 1.0
num_event = 4
reg_factor = 10.0
intra_loss_weight = [1.0, 2.0]


batch_size = 64
num_epochs = 200
sigma1 = 1.0
sampling_rate = 0.7
keep_prob_rate = 0.4
#没有使用tumor_volume
clinic_vars = ['age','sex','EBV_DNA','tumor_volume','HGB','ALB','CRP','LDH','smokingcut','drinkingcut','His_cancercut','treatment','HGBcut','ALBcut','CRPcut','LDHcut','EBV_4k']
clinic_num = len(clinic_vars)
dim_interact_feature = 3*clinic_num
reg_W = tf.contrib.layers.l2_regularizer(scale=1e-3)
reg_W_out = tf.contrib.layers.l1_regularizer(scale=1e-4)

clinic_path = "/home/PH+BMC_matched_test.csv"
# Place data loading and preprocessing on the cpu
with tf.device('/cpu:0'):
    clinic_msag = pd.read_csv(clinic_path, header = 0, index_col = 0)
    tra_msag = clinic_msag[clinic_msag['data_cohort']==1]
    val_msag = clinic_msag[clinic_msag['data_cohort1']==1]
    test_msag = clinic_msag[clinic_msag['data_cohort2']==1]

    tra_Pat_ID = np.array(tra_msag.index)
    tra_treat = np.array(tra_msag.loc[:, 'treatment'], np.float32)
    tra_time = np.array(tra_msag.loc[:, ['OS.time', 'DMFS.time', 'LRRFS.time','DFS.time']], np.float32)
    tra_event = np.array(tra_msag.loc[:, ['OS', 'DMFS', 'LRRFS','DFS']], np.float32)
    tra_FFS_time = np.array(tra_msag.loc[:, 'DFS.time'], np.float32)
    tra_FFS_event = np.array(tra_msag.loc[:, 'DFS'], np.float32)
    tra_FFS_event[tra_FFS_event<0.0] = 0.0
    clinic_factors = np.array(tra_msag.loc[:, clinic_vars], np.float32)


    val_Pat_ID = np.array(val_msag.index)
    val_treat = np.array(val_msag.loc[:, 'treatment'], np.float32)
    val_FFS_time = np.array(val_msag.loc[:, 'DFS.time'], np.float32)
    val_FFS_event = np.array(val_msag.loc[:, 'DFS'], np.float32)
    val_FFS_event[val_FFS_event<0.0] = 0.0
    clinic_factors_val = np.array(val_msag.loc[:, clinic_vars], np.float32)

    test_Pat_ID = np.array(test_msag.index)
    test_treat = np.array(test_msag.loc[:, 'treatment'], np.float32)
    test_FFS_time = np.array(test_msag.loc[:, 'DFS.time'], np.float32)
    test_FFS_event = np.array(test_msag.loc[:, 'DFS'], np.float32)
    test_FFS_event[test_FFS_event<0.0] = 0.0
    clinic_factors_test = np.array(test_msag.loc[:, clinic_vars], np.float32)

    # import pdb; pdb.set_trace()
    ind_all = np.array(range(len(tra_msag)))
    ind_0 = ind_all[tra_msag.loc[:,'DFS'] == -1]
    ind_1 = ind_all[tra_msag.loc[:,'DFS'] == 1]
    nn_1 = len(ind_1)
    nn_0 = len(ind_0)
    print('number of patients without the event befor sampling: %d' % nn_0)
    print('number of patients with the event befor sampling: %d' % nn_1)
    np.random.seed(0)
    if nn_1 < sampling_rate*nn_0:
        out = np.random.choice(len(ind_1),int(sampling_rate*nn_0))
        ind_1 = ind_1[out]
    else:
        out = np.random.choice(len(ind_0),int(nn_1/sampling_rate))
        ind_0 = ind_0[out]

    r0 = int(batch_size/(1+sampling_rate))
    r1 = batch_size - r0
    print('number of patients without the event after sampling: %d' % len(ind_0))
    print('number of patients with the event after sampling: %d' % len(ind_1))

    num_batchs = min(len(ind_0)//r0, len(ind_1)//r1)
    # print([r0,r1])
    print('num_epochs: %d' % num_batchs)
    
def _prepare_surv_data(surv_time, surv_event):
    surv_data_y = surv_time * surv_event
    surv_data_y = np.array(surv_data_y, np.float32)
    T = - np.abs(np.squeeze(surv_data_y))
    sorted_idx = np.argsort(T)
    
    return sorted_idx
    
def GetData(ind0, ind1):
    ind = np.hstack((ind0,ind1))
    np.random.shuffle(ind)
    input_idx = np.zeros((batch_size, 4), dtype = np.int32)
    
    input_time = tra_time[ind]
    input_event = tra_event[ind]
    
    sorted_idx = _prepare_surv_data(input_time[:,0], input_event[:,0])
    input_idx[:,0] = sorted_idx
    input_time[:,0] = input_time[sorted_idx,0]
    input_event[:,0] = input_event[sorted_idx,0]
    
    sorted_idx = _prepare_surv_data(input_time[:,1], input_event[:,1])
    input_idx[:,1] = sorted_idx
    input_time[:,1] = input_time[sorted_idx,1]
    input_event[:,1] = input_event[sorted_idx,1]
    
    sorted_idx = _prepare_surv_data(input_time[:,2], input_event[:,2])
    input_idx[:,2] = sorted_idx
    input_time[:,2] = input_time[sorted_idx,2]
    input_event[:,2] = input_event[sorted_idx,2]
    
    sorted_idx = _prepare_surv_data(input_time[:,3], input_event[:,3])
    input_idx[:,3] = sorted_idx
    input_time[:,3] = input_time[sorted_idx,3]
    input_event[:,3] = input_event[sorted_idx,3]
    
    input_x1 = clinic_factors[ind, :]
    
    treat = 0.5*tra_treat[ind]
    treat = treat.reshape((-1, 1))
    treat_out = np.ones((1, dim_interact_feature))
    treat_out = treat * treat_out

    
    return treat_out, input_x1, input_time, input_event, input_idx
    


def DeepSurv_loss(surv_time, surv_event, pat_ind, Y_hat):
    # Obtain T and E from self.Y
    # NOTE: negtive value means E = 0
    Y = surv_time * surv_event
    Y_c = tf.squeeze(Y)
    Y_hat_c = tf.squeeze(Y_hat)
    Y_hat_c = tf.gather(Y_hat_c,pat_ind)
    Y_label_T = tf.abs(Y_c)
    Y_label_E = tf.cast(tf.greater(Y_c, 0), dtype=tf.float32)
    Obs = tf.reduce_sum(Y_label_E)
    
    Y_hat_hr = tf.exp(Y_hat_c)
    Y_hat_cumsum = tf.log(tf.cumsum(Y_hat_hr))
    
    # Start Computation of Loss function
    
    # Get Segment from T
    _, segment_ids = tf.unique(Y_label_T)
    # Get Segment_max
    loss_s2_v = tf.segment_max(Y_hat_cumsum, segment_ids)
    # Get Segment_count
    loss_s2_count = tf.segment_sum(Y_label_E, segment_ids)
    # Compute S2
    loss_s2 = tf.reduce_sum(tf.multiply(loss_s2_v, loss_s2_count))
    # Compute S1
    loss_s1 = tf.reduce_sum(tf.multiply(Y_hat_c, Y_label_E))
    # Compute Breslow Loss
    loss_breslow = tf.divide(tf.subtract(loss_s2, loss_s1), Obs)
    
    return loss_breslow

def _RankLoss(surv_time, surv_event, pat_ind, Y_hat):
    """
    Y_hat: predicted value
    label: censored(1) or uncensored(-1)
    t: time to events
    """

    Y_hat_c = tf.squeeze(Y_hat)
    Y_hat_c = tf.gather(Y_hat_c,pat_ind)
    Y_hat_c = tf.reshape(Y_hat_c, [batch_size, 1])
    surv_time = tf.reshape(surv_time, [batch_size, 1])
    surv_event = tf.reshape(surv_event, [batch_size, 1])
    one_vector = tf.ones((batch_size,1),np.float32)
    I_2 = tf.cast(tf.greater(surv_event, 0), dtype = tf.float32)
    I_2 = tf.diag(tf.squeeze(I_2))
    R = tf.matmul(Y_hat_c, tf.transpose(one_vector)) - tf.matmul(one_vector, tf.transpose(Y_hat_c))
    T = tf.nn.relu(tf.sign(tf.matmul(one_vector, tf.transpose(surv_time)) - tf.matmul(surv_time, tf.transpose(one_vector))))
    # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j
    
    T = tf.matmul(I_2, T) # only remains T_{ij}=1 when event occured for subject i
    
    rank_loss = tf.math.reduce_mean(T * tf.exp(-R/sigma1), reduction_indices=[0,1], keepdims=False)
    
    return rank_loss

def exponential_decay_with_warmup(warmup_step,learning_rate_base,global_step,learning_rate_step,learning_rate_decay,staircase=False):

    with tf.name_scope("exponential_decay_with_warmup"):
        linear_increase=learning_rate_base*tf.cast((global_step+1)/warmup_step,tf.float32)
        exponential_decay=tf.train.exponential_decay(learning_rate_base,
                                                     global_step-warmup_step,
                                                     learning_rate_step,
                                                     learning_rate_decay,
                                                     staircase=staircase)
        learning_rate=tf.cond(global_step<=warmup_step,
                              lambda:linear_increase,
                              lambda:exponential_decay)
        return learning_rate

def _create_fc_layer(x, output_dim, activation, scope, keep_prob = None,use_bias=True, w_reg = None, initial_W = None):
    if initial_W is None:
        initial_W = tf.contrib.layers.xavier_initializer()
    with tf.variable_scope(scope):
        layer_out = tf.layers.dense(inputs=x, use_bias=use_bias, units=output_dim, kernel_initializer=initial_W, kernel_regularizer=w_reg)
        if keep_prob is not None:
            layer_out = tf.nn.dropout(layer_out, keep_prob=keep_prob)
        if activation == 'relu6':
            layer_out = tf.nn.relu6(layer_out)
        elif activation == 'relu':
            layer_out = tf.nn.relu(layer_out)
        elif activation == 'tanh':
            layer_out = tf.nn.tanh(layer_out)
        else:
            raise NotImplementedError('activation not recognized')
    
    return layer_out
def Get_loss(output, s_time, s_event, Pat_ind):
    #tf.gather(output, 0, axis = -1) 变成一维拉
    OS_loss_cox = DeepSurv_loss(s_time[:,0], s_event[:,0], Pat_ind[:,0], output[:,0])
    OS_loss_rank = _RankLoss(s_time[:,0], s_event[:,0], Pat_ind[:,0], output[:,0])
    DMFS_loss_cox = DeepSurv_loss(s_time[:,1], s_event[:,1], Pat_ind[:,1], output[:,1])
    DMFS_loss_rank = _RankLoss(s_time[:,1], s_event[:,1], Pat_ind[:,1], output[:,1])
    LRFS_loss_cox = DeepSurv_loss(s_time[:,2], s_event[:,2], Pat_ind[:,2], output[:,2])
    LRFS_loss_rank = _RankLoss(s_time[:,2], s_event[:,2], Pat_ind[:,2], output[:,2])

    DFS_loss_cox = DeepSurv_loss(s_time[:,3], s_event[:,3], Pat_ind[:,3], tf.reduce_max(output,1))
    DFS_loss_rank = _RankLoss(s_time[:,3], s_event[:,3], Pat_ind[:,3], tf.reduce_max(output,1))

    loss_cox = alpha*DFS_loss_cox + belta*OS_loss_cox + gamma*DMFS_loss_cox + sita*LRFS_loss_cox
    loss_rank = alpha*DFS_loss_rank + belta*OS_loss_rank + gamma*DMFS_loss_rank + sita*LRFS_loss_rank
    return loss_cox, loss_rank

def main():
        
    with tf.device('/gpu:0'):
        x = tf.placeholder(tf.float32, [None, clinic_num], name = 'input')
        s_time = tf.placeholder(tf.float32, [None,num_event], name = 'surv_time')
        s_event = tf.placeholder(tf.float32, [None,num_event], name = 'surv_event')
        Pat_ind = tf.placeholder(tf.int32, [None,num_event], name = 'Pat_ind')
        keep_prob = tf.placeholder(tf.float32, name = 'keep_rate')
        treatment = tf.placeholder(tf.float32, [None, dim_interact_feature], name = 'treatment')
        global_step = tf.placeholder(tf.int32, [])
        # model
        fc = _create_fc_layer(x, 3*clinic_num, 'relu', 'shared_layer', keep_prob, w_reg = reg_W)
        # fc = tf.concat([x,fc0], axis=1)

        # fc1_1 = _create_fc_layer(fc, 5*clinic_num, 'relu', 'specific_layer1_1', keep_prob, w_reg = reg_W)
        fc1_2 = _create_fc_layer(fc, 1*clinic_num, 'relu', 'specific_layer1_2', keep_prob, w_reg = reg_W)
        output1 = _create_fc_layer(fc1_2, num_event-1, 'tanh', 'output_1', use_bias= False, w_reg = reg_W_out)


        fc2_1 = tf.multiply(treatment, fc)
        # fc2_1 = _create_fc_layer(fc2_1, 5*clinic_num, 'relu', 'specific_layer2_1', keep_prob, w_reg = reg_W)
        fc2_2 = _create_fc_layer(fc2_1, 1*clinic_num, 'relu', 'specific_layer2_2', keep_prob, w_reg = reg_W)
        output2 = _create_fc_layer(fc2_2, 1, 'tanh', 'output_2', use_bias= False, w_reg = reg_W_out)
        # loss
        loss_cox_prog, loss_rank_prog = Get_loss(output1, s_time, s_event, Pat_ind)
        pred_DFS = tf.reduce_max(output1,axis=1)
        loss_cox_pred = DeepSurv_loss(s_time[:,3], s_event[:,3], Pat_ind[:,3], output2)
        loss_reg = tf.losses.get_regularization_loss()
        # + intra_loss_weight[1]*loss_cox_pred
        loss_total = intra_loss_weight[0]*loss_cox_prog + reg_factor*loss_reg
        # import pdb; pdb.set_trace()
        # x1 = tf.Variable([0.2,0.3,0.5],tf.float32)
        # x2 = tf.reduce_max(x1)
        learning_rate = exponential_decay_with_warmup(warmup_step,learning_rate_base,global_step,learning_rate_step,learning_rate_decay,staircase=True)
        optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = momentum, use_nesterov = True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = optimizer.minimize(loss_total)

        restore_var = [v for v in tf.trainable_variables()]
        print(restore_var)
        # import pdb;pdb.set_trace()
        saver = tf.train.Saver(max_to_keep = 10)
        # Start Tensorflow session
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.9)#设置每个GPU使用率0.7代表70%
        config = tf.ConfigProto(gpu_options = gpu_options, allow_soft_placement = True)
        with tf.Session(config = config) as sess:
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # loader = tf.train.Saver(var_list=restore_var)
            # loader.restore(sess, snapshot_dir)
            
            gsp = 0
            # Loop over number of epochs
            for epoch in range(num_epochs):
            
                # print("{} Start epoch number: {}".format(datetime.now(), epoch))
                np.random.shuffle(ind_0)
                np.random.shuffle(ind_1)
                # Initialize iterator with the training dataset
                train_risk = 0.0
                prog_risk = 0.0
                pred_risk = 0.0
                reg_risk = 0.0
                # import pdb;pdb.set_trace()
                for i in range(num_batchs):
                    gsp += 1
                    ind0 = ind_0[i*r0:(i+1)*r0]
                    ind1 = ind_1[i*r1:(i+1)*r1]
                    treat, input_x1, input_time, input_event, input_idx = GetData(ind0,ind1)
                    # import pdb;pdb.set_trace()
                    # pdfs, opt = sess.run([pred_DFS,output1], feed_dict = {global_step:gsp, treatment: treat, x: input_x1, s_time: input_time, s_event: input_event, Pat_ind: input_idx, keep_prob: 1.0})
                    # pdfs1 = sess.run(pred_DFS, feed_dict = {global_step:gsp, treatment: treat, x: input_x1, s_time: input_time, s_event: input_event, Pat_ind: input_idx, keep_prob: 1.0})
                    # print(pdfs1)
                    # print(pdfs)
                    # print(opt)
                    _, opt2, _, opt, reg_ls, prog_ls, pred_ls, total_ls, now_lr = sess.run([train_step, fc, output1, pred_DFS, loss_reg, loss_cox_prog, loss_cox_pred, loss_total, learning_rate], feed_dict = {global_step:gsp, treatment: treat, x: input_x1, s_time: input_time, s_event: input_event, Pat_ind: input_idx, keep_prob: keep_prob_rate})
                    reg_risk += reg_ls
                    train_risk += total_ls
                    prog_risk += prog_ls
                    pred_risk += pred_ls
                # import pdb;pdb.set_trace()
                reg_risk /= num_batchs
                train_risk /= num_batchs
                prog_risk /= num_batchs
                pred_risk /= num_batchs
                line = 'epoch: %d, learning rate: %.5f, tatol_loss: %.4f, reg_loss: %.4f, prognosis-cox loss: %.4f, predict-cox loss: %.4f' % (epoch + 1, now_lr, train_risk, reg_risk, prog_risk, pred_risk)
                print(line)
                with open(log_path, 'a') as f:
                    f.write(line + '\n')
                if  (epoch+1)%2 == 0:
                    tra_pred = []
                    for i in range(len(tra_treat)):
                        xd = tra_treat[i]
                        treat = np.array([xd]*dim_interact_feature)
                        opt1, Pat_pred = sess.run([output1,pred_DFS], feed_dict = {x: clinic_factors[i,:].reshape(1,clinic_num), keep_prob: 1.0, treatment: treat.reshape(1,dim_interact_feature)}) 
                        if np.max(Pat_pred)>1.0:
                            import pdb;pdb.set_trace()
                        tra_pred.append(-Pat_pred)
                    # import pdb;pdb.set_trace()
                    tra_pred = np.array(tra_pred, np.float32)
                    tra_ci_value = concordance_index(tra_FFS_time, tra_pred, tra_FFS_event)
                    line = 'train cohort, CI: %.4f, epoch: %d' % (tra_ci_value, epoch)
                    print(line)
                        
                    val_pred = []
                    for i in range(len(val_treat)):
                        xd = val_treat[i]
                        treat = np.array([xd]*dim_interact_feature)
                        opt = sess.run(pred_DFS, feed_dict = {x: clinic_factors_val[i,:].reshape(1,clinic_num), 
                                                                 keep_prob: 1.0, treatment: treat.reshape(1,dim_interact_feature)})
                        print(opt)
                        opt = sess.run(output1, feed_dict = {x: clinic_factors_val[i,:].reshape(1,clinic_num), 
                                                                 keep_prob: 1.0, treatment: treat.reshape(1,dim_interact_feature)})
                        print(opt)
                        opt1, Pat_pred = sess.run([output1,pred_DFS], feed_dict = {x: clinic_factors_val[i,:].reshape(1,clinic_num), 
                                                                 keep_prob: 1.0, treatment: treat.reshape(1,dim_interact_feature)}) 
                        print(opt1)
                        print(Pat_pred)
                        
                        opt1, Pat_pred = sess.run([output1,pred_DFS], feed_dict = {x: clinic_factors_val[i,:].reshape(1,clinic_num), 
                                                                 keep_prob: 1.0, treatment: treat.reshape(1,dim_interact_feature)}) 
                        print(opt1)
                        print(Pat_pred)
                        val_pred.append(-Pat_pred)
                    import pdb;pdb.set_trace()
                    val_pred = np.array(val_pred, np.float32)
                    val_ci_value = concordance_index(val_FFS_time, val_pred, val_FFS_event)
                    line = 'validation cohort, CI: %.4f, epoch: %d' % (val_ci_value, epoch)
                    print(line)
                        
                    test_pred = []
                    for i in range(len(test_treat)):
                        xd = test_treat[i]
                        treat = np.array([xd]*dim_interact_feature)
                        # import pdb;pdb.set_trace()
                        opt, Pat_pred = sess.run([output1,pred_DFS], feed_dict = {x: clinic_factors_test[i,:].reshape(1,clinic_num), keep_prob: 1.0, treatment: treat.reshape(1,dim_interact_feature)}) 
                        test_pred.append(-Pat_pred[0])
                    # import pdb;pdb.set_trace()
                    test_pred = np.array(test_pred, np.float32)
                    test_ci_value = concordance_index(test_FFS_time, test_pred, test_FFS_event)
                    line = 'test cohort, CI: %.4f, epoch: %d' % (test_ci_value, epoch)
                    print(line)
                    
                    

if __name__ == '__main__':
    main()