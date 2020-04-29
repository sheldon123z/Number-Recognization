import numpy as np
import random
import minist_cvs3
import matplotlib.pyplot as plot
import sys
class SoftmaxLayer:
    @classmethod
    def p_result(cls,input_z):
        exp = np.exp(input_z)
        res= np.exp(input_z) / np.sum(np.exp(input_z), axis=0, keepdims=True)
        return res

class Network:

    def __init__(self, layer, batch_size, epoch_num):
        
        self.layer = layer
        self.num_layer = len(self.layer)
        self.biases = [np.random.randn(y, 1) for y in self.layer[1:]]
        #initialize gaussian random matrix with mean 0 variance sqrt(num_of_neurons each layer)
        #this will help decrease the risk of being saturated nodes
        #如果单纯的初始化成（0，1）分布的高斯weight，有可能会产生饱和node，也就是说会有很接近-1，和1的weight参数出现，这样的参数
        #会导致后续学习进程中，参数改变的很慢，压缩方差的做法会提升高斯分布的高度，让产生概率集中在0的附近而不靠近1和-1
        self.weights = [np.random.randn(y, x)*1/np.sqrt(x) for x, y in list(zip(self.layer[:-1], self.layer[1:]))]
        
        self.batch_size = batch_size
        self.num_epoch = epoch_num


    def feed_forward(self,a):
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight,a) + bias)
        return a

    def train_network(self,training_data,training_data_length,test_data,test_img,initial_learning_rate,decay,lmbda):
        
        training_cost =[]
        training_accuracy = []
        test_accuracy =[]
        epoches =[]
        if test_data: 
            len_test = len(test_data)
        
        
        for epoch in range(self.num_epoch):
            epoches.append(epoch)
            learning_rate =  initial_learning_rate * (1/(1 + decay * epoch))
            self.train_one_epoch(training_data,training_data_length,learning_rate,lmbda)
            print ("Epoch {} training complete".format(epoch))
            accuracy = self.cal_training_accuracy(training_data,True)
            training_accuracy.append(accuracy * 100/training_data_length)
            print ("Accuracy on training data: {} / {}".format(
                accuracy, training_data_length))  
            
            # cost = self.total_cost(training_data,lmbda) 
            # print("cost on training data: {}".format(cost))
            # training_cost.append(cost)
            
            if test_data:
                print ("Accuracy on testing Epoch {0}: {1} / {2}".format(
                        epoch, self.cal_test_accuracy(test_data), len_test))
                
                 
            if test_img:
                self.predict_result(test_img,True)
                
        fig = plot.figure()
        plt1 = fig.add_subplot(121,xlabel = 'epoches',ylabel='accuracies',xlim = (0,10),ylim=(80,100))
        plt1.set_title('accuracy vs epoch')
        plt1.plot(epoches,training_accuracy)
        plt1.plot(epoches,training_accuracy, "o")
        
        # plt2 = fig.add_subplot(122,xlabel = 'epoches',ylabel='test data cost')
        # plt2.set_title('test datacost vs epoch')
        # plt2.plot(epoches,training_cost)
        # plt2.plot(epoches,training_cost, "o")
        plot.show()
        
    #separate the data into batches
    def separate_data(self, data,n):
        random.shuffle(training_data)
        return [ data[start:start + self.batch_size] for start in range(0, n, self.batch_size)]


    def train_one_epoch(self,training_data,training_data_length,learning_rate,lam):
        
        n = training_data_length
        baches = self.separate_data(training_data,n)

        for bach in baches:
            #define two empty new array for biases and weights
            delta_biases = []
            delta_weights = []
            for bias in self.biases:
                delta_biases.append(np.zeros(bias.shape))
            
            for weight in self.weights:
                delta_weights.append(np.zeros(weight.shape))
            
            for X, y in bach:
                delta_biase,delta_weight = self.back_propagation(X,y)
                
                delta_weights = [ nw+dw for nw, dw in zip(delta_weights,delta_weight)]
                delta_biases = [ nb+db for nb, db in zip(delta_biases,delta_biase)]

            #update weights and biases
            new_weights=[] 
            new_biases=[]
            #gredient decent
            for d_weight, old_weight in zip(delta_weights, self.weights):
                #L2 regulation with lambda hyper parameter
                new_weights.append((1-learning_rate*(lam/n)) * old_weight - (learning_rate/self.batch_size)*d_weight)
            self.weights = new_weights
            
            #update biases
            for d_bias, old_bias in zip(delta_biases,self.biases):
                new_biases.append(old_bias-learning_rate/self.batch_size * d_bias)
            self.biases = new_biases
            

    def back_propagation(self,X,y):
    
        #feed forward implementation
        d_biases = [np.zeros(bias.shape) for bias in self.biases]
        d_weights = [np.zeros(weight.shape) for weight in self.weights]

        a = X
        inputs = []
        activations = [a]

        for bias, weight in zip(self.biases, self.weights):
            # z = weight * x + bias
            input_z = np.dot(weight,a) + bias
            # a = sigmoid(z)
            a = sigmoid(input_z)
            inputs.append(input_z)
            activations.append(a)

        
        #the difference between true value and predicted value
        #!Uncomment the line to use sigmoid output layer
        #!if use the sigmoid output layer, the accuracy calculation must be changed
        #use sigmoid layer as the output layer
        # output_activation = activations[-1]
        # diff = (output_activation - y) * d_sigmoid(inputs[-1])

        #! softmax layer
        # softmax layer loss 
        #the softmax layer output diff for cross entropy cost function 
        #softmax derivative as per 'a' is f' = a(1-a)
        #the cross entropy derivative (a-y)/a(a-a)  
        #mutiply two derivatives get the output diff is a-y
        soft_layer= SoftmaxLayer()
        soft_out = soft_layer.p_result(inputs[-1])
        activations[-1] = soft_out
        diff = (soft_out - y) 
        
        
        d_biases[-1] = diff
        d_weights[-1] = np.dot(diff,activations[-2].transpose())


        #calculate from the second layer's activation 
        for layer in range(2, self.num_layer):
            z = inputs[-layer]
            #BP2 An equation for the error δl in terms of the error in the next layer
            diff = np.dot(self.weights[-layer+1].transpose(), diff) * d_sigmoid(z)
            #BP3 An equation for the rate of change of the cost with respect to any bias in the network
            d_biases[-layer] = diff
            #BP4 An equation for the rate of change of the cost with respect to any weight in the network: 
            d_weights[-layer] = np.dot(diff, activations[-layer-1].transpose())

        return (d_biases, d_weights)
    
    #calculate the training data set accuracy
    def cal_training_accuracy(self,training_data,softmax=True):
        results =[]
        for (x, y) in training_data:
            a = x
            if softmax:
                i = 1
                for bias, weight in zip(self.biases, self.weights):
                    if i != self.num_layer:
                        a = sigmoid(np.dot(weight,a) + bias)
                    else:
                        a = SoftmaxLayer.p_result(a)
            else:           
                for bias, weight in zip(self.biases, self.weights):
                    a = sigmoid(np.dot(weight,a) + bias)
                         
            #append the predicted result and real result pair            
            results.append((np.argmax(a),np.argmax(y)))
        
        correct_num = sum(int(x == y) for (x, y) in results)
        return correct_num

    #calculate the accuracy of test data set 
    def cal_test_accuracy(self, test_data,softmax=True):
        result = []
        zs = []
        accuracy = 0
        for (x,y) in test_data:
            a = x
            if softmax:
                i = 1
                for bias, weight in zip(self.biases, self.weights):
                    if i != self.num_layer:
                        a = sigmoid(np.dot(weight,a) + bias)
                    else:
                        a = SoftmaxLayer.p_result(a)
            else:           
                for bias, weight in zip(self.biases, self.weights):
                    a = sigmoid(np.dot(weight,a) + bias)
                    
            prediction = np.argmax(a)
            result.append((prediction,y))

        for (z,y) in result:
            if z == y:
                accuracy +=1
            zs.append(z)
            
        return accuracy
    
    #output the predicted number labels to cvs file
    #set false if use sigmoid output layer
    def predict_result(self, test_data,softmax):
        results = []
    
        for x in test_data:
            a = x
            if softmax:
                i = 1
                for bias, weight in zip(self.biases, self.weights):
                    if i != self.num_layer:
                        a = sigmoid(np.dot(weight,a) + bias)
                    else:
                        a = SoftmaxLayer.p_result(a)
            else:
                for bias, weight in zip(self.biases, self.weights):
                    a = sigmoid(np.dot(weight,a) + bias)
            results.append(np.argmax(a))
            
        ans = np.array(results, dtype= int)
        #save to cvs file
        np.savetxt('test_predictions.csv',ans,fmt='%d')
        
    # def total_cost(self, data, lmbda, convert=False,softmax=False):
    #     cost = 0.0
    #     for x, y in data:
    #         a = x
    #         if softmax:
    #             i = 1
    #             for bias, weight in zip(self.biases, self.weights):
    #                 if i != self.num_layer:
    #                     a = sigmoid(np.dot(weight,a) + bias)
    #                 else:
    #                     a = SoftmaxLayer.p_result(a)
    #         else:           
    #             for bias, weight in zip(self.biases, self.weights):
    #                 a = sigmoid(np.dot(weight,a) + bias)
                    
    #         if convert: y = vectorized_result(y)
    #         fn = (np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a))))
    #         cost += fn/len(data)
            
    #     cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)
    #     return cost
#-------------------------------sigmoid functions------------------------------------#

#derivate of sigmoid function    
def d_sigmoid(X):
    return (1-sigmoid(X)) * sigmoid(X)


#sigmoid function
def sigmoid(X):
    return 1.0 / (1.0 + np.exp(-X))  

def vectorized_result(j):
    j = int(j)
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

#-------------------------------read data from cvs files------------------------------------#
def read_data(training_img,training_label,test_img,test_label):
    training_input = []
    training_result = []
    test_input = []
    test_result = []
    with open(training_img,'r') as t_img:
        lines = t_img.readlines()
        #a simplified statement
        # training_input = [np.reshape(np.fromstring(line, dtype='float128', sep=','),(784,1)) for line in lines]
        for line in lines:
            image = np.fromstring(line, dtype='float128', sep=',')
            training_input.append(np.reshape(image,(784,1))/256)
            
    with open(training_label,'r') as t_lab:
        lines = t_lab.readlines()
        e = np.zeros((10,1))
        training_result = [ vectorized_result(line) for line in lines ] 
        
    training_data = list(zip(training_input,training_result)) 
    
    with open(test_img,'r') as test_img:
        lines = test_img.readlines()
        for line in lines:
            image = np.fromstring(line, dtype='float128', sep=',')
            test_input.append(np.reshape(image,(784,1))/256)
            
    if test_label:        
        with open(test_label,'r') as test_label:
            lines = test_label.readlines()
            for line in lines:
                test_result.append(int(line))
    
    if test_label:
        test_data = list(zip(test_input,test_result))
    else:
        test_data = test_input
    
    return training_data,test_data


#!enable the following 3 lines to train the full data set 
training_data, validation_data, test_data = minist_cvs3.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)

# input other data from terminal uncomment the following 8lines and comment the above 3 lines, switch the train_network parameter test_data with flase
# t_img = sys.argv[1]
# t_lab = sys.argv[2]
# test_img = sys.argv[3]

# t_img = "train_image.csv"
# t_lab = "train_label.csv"
# test_img = "test_image.csv"
# test_lab = "test_label.csv"
# training_data, test_data = read_data(t_img,t_lab,test_img,test_lab)

print("total num of training data is {}".format(len(training_data)))
training_data_length = len(training_data)
#change the array to change the structure of the network
#[784,30,20,10] means the network has input layer 784 nodes, 2 hidden layers for 30, 20 neurons respectively,
# and 10 output neurons output layer
#second parameter is the mini batch size, last parameter is the epochs you want to train
network = Network([784,30,10],100,10)
network.train_network(training_data,
                      training_data_length,
                      test_data,
                      False,
                      initial_learning_rate = 0.5,
                      decay = 0.1,
                      lmbda = 5)



        

        

            








        
            

