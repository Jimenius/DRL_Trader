import tensorflow as tf
import numpy as np
import time
from matplotlib import pyplot as plt

class norm_layer(object):
    def __init__(self, input_x):
        """
        :param input_x: The input that needed for normalization.
        """
        with tf.variable_scope('batch_norm'):
            mean, variance = tf.nn.moments(input_x, axes=[0], keep_dims=True)
            cell_out = tf.nn.batch_normalization(input_x,
                                                 mean,
                                                 variance,
                                                 offset=np.zeros_like(input_x),
                                                 scale=np.ones_like(input_x),
                                                 variance_epsilon=1e-6,
                                                 name=None)
            self.cell_out = cell_out

    def output(self):
        return self.cell_out

class fc_layer(object):
    def __init__(self, input_x, in_size, out_size, rand_seed, activation_function=None, index=0):
        with tf.variable_scope('fc_layer_%d' % index):
            with tf.name_scope('fc_kernel'):
                w_shape = [in_size, out_size]
                weight = tf.get_variable(name='fc_kernel_%d' % index, shape=w_shape,
                                         initializer=tf.glorot_normal_initializer(seed=rand_seed))
                self.weight = weight

            with tf.variable_scope('fc_kernel'):
                b_shape = [out_size]
                bias = tf.get_variable(name='fc_bias_%d' % index, shape=b_shape,
                                       initializer=tf.glorot_uniform_initializer(seed=rand_seed))
                self.bias = bias

            cell_out = tf.add(tf.matmul(input_x, weight), bias)
            if activation_function is not None:
                cell_out = activation_function(cell_out)

            self.cell_out = cell_out

            tf.summary.histogram('fc_layer/{}/kernel'.format(index), weight)
            tf.summary.histogram('fc_layer/{}/bias'.format(index), bias)

    def output(self):
        return self.cell_out

def ValueNet(input_x,input_y,input_size=30, output_size=3,fc_units=[84], l2_norm=0.01, seed=235):

    fc_layer_0 = fc_layer(input_x=input_x,
                          in_size=input_size,
                          out_size=fc_units[0],
                          rand_seed=seed,
                          activation_function=tf.nn.softplus,
                          index=0)
    
    #nm_layer_0 = norm_layer(input_x=fc_layer_0.output())

    fc_layer_1 = fc_layer(input_x=fc_layer_0.output(),
                          in_size=fc_units[0],
                          out_size=fc_units[1],
                          rand_seed=seed,
                          activation_function=tf.nn.relu,
                          index=1)

    #nm_layer_1 = norm_layer(input_x=fc_layer_1.output())

    fc_layer_2 = fc_layer(input_x=fc_layer_1.output(),
                          in_size=fc_units[1],
                          out_size=output_size,
                          rand_seed=seed,
                          activation_function=None,
                          index=2)
    
    Out=fc_layer_2.output()
    fc_w = [fc_layer_0.weight, fc_layer_1.weight, fc_layer_2.weight]

    # loss
    with tf.name_scope("loss"):
        l2_loss = tf.reduce_sum([tf.norm(w) for w in fc_w])
        cross_entropy_loss = tf.reduce_sum((Out-input_y)**2,name='cross_entropy')
        loss = tf.add(cross_entropy_loss, l2_norm * l2_loss, name='loss')

        tf.summary.scalar('ValueNet_loss', loss)
    
    return Out, loss

def cross_entropy(output, input_y):
    with tf.name_scope('cross_entropy'):
        label = tf.one_hot(input_y, 3)
        ce = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output))

    return ce


def train_step(loss, learning_rate=1e-3):
    with tf.name_scope('train_step'):
        step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return step

def LegalEpsilonGreedy(q=[0,0,1],epsilon=0.1, pAction=0, Balance = [0,1e3], Price = 0):
    assert len(q)==3
    p=np.zeros(3)
    d = {}
    for i in range(3):
        d[i-1] = q[i]
    
    if Balance[1] < Price * 2:
        d.pop(1)

    if Balance[0] <= 0:
        d.pop(-1)

    if pAction != 0 and -pAction in d:
        d.pop(-pAction)

    if len(d) == 1:
        return 0

    maxpos = max(d, key = lambda k: d[k])
    for i in d:
        if i==maxpos:
            p[i+1]=1-epsilon
        else:
            p[i+1]=epsilon/(len(d) - 1)

    return np.random.choice([-1,0,1],size=1,p=p)
    raise NotImplementedError

def training(X,Price,
             X_test,Price_test,
             fc_units=[84],
             l2_norm=0.01,
             seed=235,
             input_size=35,
             learning_rate=1e-2,
             epoch=100,
             epsilon=0.1,
             yita=0.90,
             TransactionRecord=False,
             verbose=False,
             InitialBalance=[0,1e6]):
    print("Building Parameters: ")
    print("fc_units={}".format(fc_units))
    print("l2_norm={}".format(l2_norm))
    print("seed={}".format(seed))
    print("learning_rate={}".format(learning_rate))

    BestReward=-999999999
    
    # define the variables and parameter needed during training
    with tf.name_scope('inputs'):
        xs = tf.placeholder(shape=[1,X.shape[0]+2], dtype=tf.float32)
        ys = tf.placeholder(shape=[3], dtype=tf.float32)

    output, loss = ValueNet(input_x=xs,input_y=ys,
                            input_size=input_size,output_size=3,
                            fc_units=fc_units,l2_norm=l2_norm,seed=seed)

    step = train_step(loss,learning_rate=learning_rate)

    iter_total = 0
    cur_model_name = 'Model_{}'.format(int(time.time()))
    
    with tf.Session() as sess:
        merge = tf.summary.merge_all()

        writer = tf.summary.FileWriter("log/{}".format(cur_model_name), sess.graph)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        RecordLength=X.shape[1]
        
        alpha = 0.5
        beta = 1
        
        for epc in range(epoch):
            print("epoch {} ".format(epc + 1))
            
            iter_total=0
            illegal=False
            
            ActionRecord=np.zeros(RecordLength)
            HoldingRecord=np.zeros(RecordLength)
            BalanceRecord=np.zeros(RecordLength)
            
            Date=0
            Balance=[0, 1e3]
            oldasset = 1e3
            asset = 1e3
            pAction = 0
            #print Balance
            CurrentState=np.concatenate((X[:,Date],Balance))
            qValue=sess.run([output], feed_dict={xs: np.array([CurrentState]), ys: np.array([0,0,0])})
            Action=LegalEpsilonGreedy(qValue[0][0],epsilon, pAction, Balance, Price[0])
            
            while Date<RecordLength-1:
                
                iter_total += 1
                
                OldDate=Date
                
                #Calculate the reward

                if Date == 0:
                    dPrice = 0
                else:
                    dPrice = alpha * (Price[Date] - Price[Date - 1])

                if Action == 0:
                    Date += 1
                    asset=Balance[0] * Price[Date] + Balance[1]
                    reward = -10
                    pAction = 0
                elif Action==1:
                    if Balance[1]<Price[Date]:
                        reward=-100*Price[Date]
                        #Balance=Balance
                        Date+=1
                        illegal=True
                    else:
                        Balance[0]+=1
                        Balance[1]-=Price[Date]
                        pAction = 1
                        asset=Balance[0] * Price[Date] + Balance[1]
                        reward = asset - oldasset + dPrice + beta * (30 - Balance[0])
                elif Action==-1:
                    if Balance[0]<=0:
                        reward=-20*Price[Date]
                        Balance=Balance
                        Date+=1
                        illegal=True
                        pAction = 0
                    else:
                        Balance[0]-=1
                        Balance[1]+=Price[Date]
                        pAction = -1
                        asset=Balance[0] * Price[Date] + Balance[1]
                        reward = asset - oldasset - dPrice + beta * (Balance[0] - 10)
                    
                    #print Action
                if TransactionRecord:
                    ActionRecord[OldDate]+=Action
                    HoldingRecord[OldDate]=Balance[0]
                    BalanceRecord[OldDate]=Balance[1]
                
                #End of Calculating reward                
                
                OldState=CurrentState
                OldAction=Action
                OldqValue=qValue[0][0]
                if Date != OldDate:
                    oldasset = asset
                
                CurrentState=np.concatenate((X[:,Date],Balance))
                qValue=sess.run([output], feed_dict={xs: np.array([CurrentState]), ys: np.array([0,0,0])})
                Action=LegalEpsilonGreedy(q=qValue[0][0],epsilon =epsilon, pAction=pAction, Balance=Balance, Price=Price[OldDate])
                
                #Back Propagation
                
                Coef=np.zeros(3)
                DeltaGain=reward+yita*qValue[0][0][Action]
                for i in range(3):
                    if i==OldAction:
                        Coef[i]=DeltaGain
                    else:
                        Coef[i]=OldqValue[i]
                _, cur_loss=sess.run([step, loss], feed_dict={xs: np.array([OldState]), ys: Coef})
                
                #End of Back Propagation


                
                """
                Testing Part
                """
                if iter_total % 2000000 == 0:
                    
                    #Trading
                    Test_RecordLength=X_test.shape[1]
                    Test_Date=0
                    Test_Balance=[0,1e3]
                    Test_Reward=0
                    
                    illegal = False
                    TA = np.zeros(Test_RecordLength) ##TestAction
                    TH = np.zeros(Test_RecordLength) ##TestHolding
                    TB = np.zeros(Test_RecordLength) ##TestBalance
                    TOA = 1e3 ##TestOldAsset
                    TAss = 0 ##TestAsset
                
                    while Test_Date<Test_RecordLength:
                        Test_State=np.concatenate((X_test[:,Test_Date],Test_Balance))
                        Test_qValue=sess.run([output], feed_dict={xs: np.array([Test_State]), ys: np.array([0,0,0])})
                        Test_Action=np.argmax(Test_qValue[0][0])-1
                        
                        TOD = Test_Date
                        if Test_Action==1:
                            if Test_Balance[1]<Price_test[Test_Date]:
                                #Test_Reward+=-1000*Price_test[Test_Date]
                                #Test_Balance=Test_Balance
                                Test_Action = 0
                                Test_Date+=1
                                illegal=True
                            else:
                                TAss = Test_Balance[0] * Price_test[Test_Date] + Test_Balance - TOA
                                Test_Reward -= Price_test[Test_Date]
                                Test_Balance[0]+=1
                                Test_Balance[1]-=Price_test[Test_Date]
                        elif Test_Action==0:
                            Test_Reward+=0
                            Test_Balance=Test_Balance
                            Test_Date=Test_Date+1
                        elif Test_Action==-1:
                            if Test_Balance[0]<=0:
                                #Test_Reward+=-1000*Price_test[Test_Date]
                                #Test_Balance=Test_Balance
                                Test_Action = 0
                                Test_Date+=1
                                illegal=True
                            else:
                                Test_Reward+=Price_test[Test_Date]
                                Test_Balance[0]-=1
                                Test_Balance[1]+=Price_test[Test_Date]
                    
                        if TransactionRecord:
                            TA[TOD]+=Test_Action
                            TH[TOD]=Test_Balance[0]
                            TB[TOD]=Test_Balance[1]
                    
                        if verbose:
                            print('Iteration Time: {}, Final Reward: {}'.format(iter_total,Test_Reward))
                    
                    if Test_Reward>BestReward:
                        print('Get Best Reward! The Reward is {}, In Epoch {}, Day {}, illegal record={}'
                                  .format(Test_Reward,epc+1,Date+1,illegal))
                        BestReward=Test_Reward
                        saver.save(sess, 'model/{}'.format(cur_model_name))
                    '''
                    if TransactionRecord:
                        plt.subplot(411)
                        plt.plot(Price_test)
                        plt.ylabel('Price')
                        plt.subplot(412)
                        plt.plot(TA)
                        plt.ylabel('Action')
                        plt.subplot(413)
                        plt.plot(TH)
                        plt.ylabel('Holding')
                        plt.subplot(414)
                        plt.plot(TB)
                        plt.ylabel('Balance')
                        plt.show()
                        print('The Reward is {}, In Epoch {}, Day {}, illegal record={}'
                        .format(Test_Reward,epc+1,Date+1,illegal))
                        
            '''
            if TransactionRecord:
                ActionRecord[RecordLength - 1] = 0
                HoldingRecord[RecordLength - 1] = HoldingRecord[RecordLength - 2]
                BalanceRecord[RecordLength - 1] = BalanceRecord[RecordLength - 2]
                plt.figure(figsize = (10,8))
                earning = 'Total earning = ' + str(Balance[0] * Price[RecordLength - 1] + Balance[1] - 1e3)
                plt.subplot(411)
                plt.plot(Price)
                plt.ylabel('Price')
                plt.title(earning)
                plt.subplot(412)
                plt.plot(ActionRecord)
                plt.ylabel('Action')
                plt.subplot(413)
                plt.plot(HoldingRecord)
                plt.ylabel('Holding')
                plt.subplot(414)
                plt.plot(BalanceRecord)
                plt.ylabel('Balance')
                
                filename = 'Epoch_' + str(epc + 1) + '.png'
                plt.savefig(filename)
                
                print earning
                #print epc
                #if epc == 0:
                    #plt.savefig('Typical result')
                    
    print("Training ends. Got the best reward {}, the name of model is {}".format(BestReward,cur_model_name))
