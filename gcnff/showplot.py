import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def showfig(filename):
    with open(filename,'r') as f:
        try:
            a,b,c=f.readline().split()
            print("\n\tIs showing the changes of erros with epoches:\n\n")
            epoch=list()
            train_error=list()
            vali_error=list()
            while 1:
                try:
                    epoches,train_errors,vali_errors=f.readline().split()
                    epoch.append(int(epoches))
                    train_error.append(float(train_errors))
                    vali_error.append(float(vali_errors))
                except:
                    break
            fig=plt.subplot(1,1,1)
            fig.plot(train_error,label='Training Error')
            fig.plot(vali_error,label='Validation Error')
            fig.legend()
            fig.set_xlabel('Epoches',fontsize=10)
            fig.set_ylabel('Errors',fontsize=10)
            fig.figure.savefig('./Epoches-Errors.png')
        except:
            a,b=f.readline().split()
            a=float(a)
            b=float(b)
            print("\n\tComparing predicted and true values:\n")
            pred=list()
            true=list()
            pred.append(a)
            true.append(b)
            while 1:
                try:
                    a,b=f.readline().split()
                    pred.append(float(a))
                    true.append(float(b))
                except:
                    break
            if max(pred)>max(true):
                max_num=max(pred)
            else:
                max_num=max(true)
            if min(pred)<min(true):
                min_num=min(pred)
            else:
                min_num=min(true)

            fig=plt.subplot(1,1,1)
            fig.plot(np.linspace(min_num,max_num),np.linspace(min_num,max_num),linewidth=2,color=[0,0,0,1],zorder=0)
            fig.scatter(pred,true)
            fig.set_xlabel('Predict',fontsize=10)
            fig.set_ylabel('True',fontsize=10)
            fig.figure.savefig('./Pred-True.png')
