Welcome to optimisation program 101 !

#Classic libraries
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl
import pandas as pd

#Libraries for machine learning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve, validation_curve
from sklearn import preprocessing

#Libraries for statistics (ANOVA)
import statsmodels.api as sm
from statsmodels.formula.api import ols

##############################################

#Function that makes the ANOVA of the tested parameters for a defined studied result and print it as a histogram

#stud_var is a string ('') corresponding to the studied result
def anova(names, stud_var):
    stud_var += ' ~'
    for i in range(len(names)):
        stud_var+= ' + ' + 'C(' + names[i] + ')'
    print(stud_var)
    model = ols(stud_var, df).fit()
    model.summary()
    res = sm.stats.anova_lm(model, typ= 1)   # Need type 1 to agree with paper and with Matlab
    print(res)
    objects=names
    objects=tuple(objects)
    y_pos = np.arange(len(objects))  # this just makes an array [0, 1, 2, 3]
                                     # arrange makes evenly spaced values on a 
                                     # given interval.  Sort of expects integers

    totalSSRnoRes = sum(res.sum_sq)-res.sum_sq[-1]  # for normalizing

    performance = []
    for i in range(len(objects)):
        performance+=[res.sum_sq[i]/totalSSRnoRes]

    plt.figure()                   # can number them but they will not overwrite unless you close them

    if len(objects)==3:
        colour=['skyblue', 'peru', 'yellowgreen']
    elif len(objects)==2:
        colour=['skyblue', 'peru']
    elif len(objects)==4:
        colour=['skyblue', 'peru', 'yellowgreen', 'gold']
    elif len(objects)==5:
        colour=['skyblue', 'peru', 'yellowgreen', 'gold', 'pink']

    plt.bar(y_pos, performance, 
            align='center', 
            width=0.8,              # default is 0.8
            alpha=1.0,              # this is transparency, 1.0 is solid
            color=colour)
    plt.xticks(y_pos, objects)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    font = {'size': 18}
    plt.xlabel('Conditions', font)
    plt.ylabel('Fraction of total variance', font)
    plt.title('Anova', font)


    plt.show()

###############################################

#Function that creates and fits the model

#define df et names before hand
#stud_var is a string ('')
#size is an array [] giving the range of the obtained results
def pred_fit(stud_var, size):

#Optimistion of C:
    variable = df.loc[:, names]
    target = df[stud_var]
    param_range = np.linspace(2,200,50)   # set the range for parameter C
    train_loss, test_loss = validation_curve(
            svm.SVR(kernel='rbf', gamma=0.5), 
            preprocessing.scale(variable), preprocessing.scale(target.values.ravel()), param_name='C',
            param_range=param_range, cv=10, 
            scoring = 'neg_mean_squared_error')
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    plt.figure(1)
    plt.plot(param_range, train_loss_mean, 'o-', color="r", label="Training")
    plt.plot(param_range, test_loss_mean, 'o-', color="g", label="Cross_validation")
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    font = {'size': 22}
    plt.xlabel("C", font)
    plt.ylabel("Loss", font)
    plt.legend(loc="best", fontsize=22)
    plt.show()
    label=[]
    verif=[]
    for i in range(len(param_range)):    
        label+=[[test_loss_mean[i], param_range[i]]]
        verif+=[[train_loss_mean[i], param_range[i]]]
    C=min(label)[1]
    a=label.index(min(label))
    if C<100:
        C=100

#Optimistion of gamma:
    param_range = np.linspace(0,1,50) # set the range for parameter gamma
    train_loss, test_loss = validation_curve(
             svm.SVR(kernel='rbf', C=C),
             preprocessing.scale(variable), preprocessing.scale(target.values.ravel()),
             param_name='gamma',
             param_range=param_range, cv=10,
             scoring = 'neg_mean_squared_error')
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    label=[]
    for i in range(len(param_range)):    
        label+=[[test_loss_mean[i], param_range[i]]]
    gamma=min(label)[1]
    a=label.index(min(label))
    if a<len(label)-1:
        if label[a][0] < label[a+1][0]-0.1:
            del label[a]
        gamma=min(label)[0]
    if gamma==0:
        gamma=0.1
    plt.figure(2)
    plt.plot(param_range, train_loss_mean, 'o-', color="r", label="Training")
    plt.plot(param_range, test_loss_mean, 'o-', color="g", label="Cross_validation")
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    font = {'size': 22}
    plt.xlabel("gamma", font)
    plt.ylabel("Loss", font)
    plt.legend(loc="best", fontsize=22)
    plt.show()

#optimisation of epsilon
    param_range = np.linspace(0,1,50) # set the range for parameter gamma
    train_loss, test_loss = validation_curve(
         svm.SVR(kernel='rbf', C=C, gamma=gamma),
         preprocessing.scale(variable), preprocessing.scale(target.values.ravel()),
         param_name='epsilon',
         param_range=param_range, cv=10,
         scoring = 'neg_mean_squared_error')
    train_loss_mean = -np.mean(train_loss, axis=1)
    test_loss_mean = -np.mean(test_loss, axis=1)
    label=[]
    verif=[]
    epsilon=0
    for i in range(len(param_range)):    
        label+=[[test_loss_mean[i], param_range[i]]]
        verif+=[[train_loss_mean[i], param_range[i]]]
        if verif[i][0]>0.0001 and verif[i-1][0]<0.0001:
            epsilon=verif[i][1]
    plt.figure(3)
    plt.plot(param_range, train_loss_mean, 'o-', color="r", label="Training")
    plt.plot(param_range, test_loss_mean, 'o-', color="g", label="Cross_validation")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    font = {'size':15}
    plt.xlabel("epsilon")
    plt.ylabel("Loss")
    plt.legend(loc="best",fontsize=15)
    plt.show()

#fitting function creation:
    reg_stud_var = Pipeline(steps = [('scl', StandardScaler()), ('clf', svm.SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon))])
    reg_stud_var.fit(variable, target) #fit the fundtion to the values.
    calc_var= stud_var + '_pred_svm'
    df[calc_var] = reg_stud_var.predict(variable) # crée une colonne dans le tableau avec les nouvelles valeurs prédites

#graph plotting:
    fig, ax1 = plt.subplots(1, 1, clear=True, num='PCE_pred', figsize=(5, 4))
    for label, data in df.groupby('exp'):
        plt.plot(stud_var, calc_var,'o', color=data['color'].iloc[0], data=data, label=label)
       # plt.legend()
    plt.autoscale(enable=True)
    plt.plot(size, size, ls="--", c=".3") #Trace la courbe x=y. Adapter les valeurs dans les [] aux valeurs utilisées.
    ylabel = 'Predicted ' + stud_var + ' (%)'
    xlabel = 'Mesured ' + stud_var + ' (%)'
    ax1.set_ylabel(ylabel)
    ax1.set_xlabel(xlabel)
    plt.tight_layout()

#Linear regration and r2 calculation
    X=df[[stud_var]]
    y=df[[calc_var]]
    modeleReg=LinearRegression()
    modeleReg.fit(X, y)
    modeleReg.score(X,y)
    r2=modeleReg.score(X,y)
    plt.show()
    print('R2 =', r2)
    print('C=', C)
    print('gamma=', gamma)
    print('epsilon=', epsilon)
    return reg_stud_var
    
#############################

#Function that extrapolates the results using the fitted model

#df, names, data_v and data_u to be defined before hand.
#fonc is the fitting function optimised in the previous program.
def surface_plot(fonc, xmin, xmax, ymin, ymax, vmin, vmax, colour):
    ## Optimisation of 2 parameters
    if len(names)==2:
        x_len, y_len = 100, 100
        xs = np.linspace(xmin, xmax, x_len)
        ys = np.linspace(ymin, ymax, y_len)
        xi, yi = names[0], names[1]
        fig, axs = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=True, clear=True, num='rbf pce plot', figsize=(10, 10))
        xm, ym = np.meshgrid(xs,ys)
        r = np.c_[xm.flatten(), ym.flatten()]
    #compute the values from the fit
        c = fonc.predict(r).reshape(x_len, y_len)

	#Make a contour map
        cmap = axs.contour(xs, ys, c, vmin=vmin, vmax=vmax, cmap='gray_r')
        plt.clabel(cmap, inline=1, fontsize=10)
	# Make a value map
        pmap = axs.pcolormesh(xs, ys, c, shading='gouraud', vmin=vmin,	vmax=vmax, cmap=colour)
        for label,data in df.groupby('exp'):
            axs.plot(names[0], names[1],'o',color=data['color'].iloc[0], data=data.iloc[0], mec='k', mew=0.5, label=label)
        axs.set_ylabel(f'{yi}')
        axs.set_xlabel(f'{xi}')
    
    ## Optimisation of 3 parameters
    if len(names)==3:
        v_len = len(data_v)
        vs = data_v
        x_len, y_len = 100, 100
        xs = np.linspace(xmin, xmax, x_len)
        ys = np.linspace(ymin, ymax, y_len)
        vi, xi, yi = names[0], names[1], names[2]
        fig, axs = plt.subplots(nrows=1, ncols=v_len, sharex=True, sharey=True, clear=True, num='rbf pce plot', figsize=(13, 4))
        for ax, v in zip(axs, vs):
            xm, ym = np.meshgrid(xs,ys)
            vm = v * np.ones_like(xm)
            r = np.c_[vm.flatten(), xm.flatten(), ym.flatten()]
    
	#compute the values from the fit
            c = fonc.predict(r).reshape(x_len, y_len)

	#Make a contour map
            cmap = ax.contour(xs, ys, c, vmin=vmin, vmax=vmax, cmap='gray_r')
            plt.clabel(cmap, inline=1, fontsize=10)
	# Make a value map
            pmap = ax.pcolormesh(xs, ys, c, shading='gouraud', vmin=vmin,	vmax=vmax, cmap=colour)
	# Plot the experimental points
            for label,data in df.query(vi + ' ==@v').groupby('exp'):
                ax.plot(names[1], names[2],'o',color=data['color'].iloc[0], data=data.iloc[0], mec='k', mew=0.5, label=label)
            ax.set_ylabel(f'{yi}')
            ax.set_xlabel(f'{xi} @ {vi} ={v:.2f}')
    
    ## Optimisation of 4 parameters
    elif len(names)==4:
        v_len = len(data_v)
        u_len= len(data_u)
        vs = data_v
        us = data_u
        x_len, y_len = 100, 100
        xs = np.linspace(xmin, xmax, x_len)
        ys = np.linspace(ymin, ymax, y_len)
        xi, yi, ui, vi = names[0], names[1], names[2], names[3]
        fig, axs = plt.subplots(nrows=u_len, ncols=v_len, sharex=True, sharey=True, clear=True, num='rbf pce plot', figsize=(13, 4*u_len))
        for u, i in zip(us, range(u_len)):    
            for ax, v in zip(axs[i], vs):
                xm, ym = np.meshgrid(xs,ys)
                vm = v * np.ones_like(xm)
                um = u * np.ones_like(xm)
                r = np.c_[xm.flatten(), ym.flatten(), um.flatten(), vm.flatten()]
	#compute the values from the fit
                c = fonc.predict(r).reshape(x_len, y_len)

	#Make a contour map
                cmap = ax.contour(xs, ys, c, vmin=vmin, vmax=vmax, cmap='gray_r')
                plt.clabel(cmap, inline=1, fontsize=10)
	# Make a value map
                pmap = ax.pcolormesh(xs, ys, c, shading='gouraud', vmin=vmin,	vmax=vmax, cmap=colour)
	# Plot the experimental points
                for label,data in df.query(vi + ' ==@v' + ' and ' + ui + '==@u').groupby('exp'):
                    ax.plot(names[0],names[1],'o',color=data['color'].iloc[0], data=data.iloc[0], mec='k', mew=0.5, label=label)
                ax.set_ylabel(f'[{yi}] // [{ui}]={u:.2f}')
                ax.set_xlabel(f'[{xi}] // [{vi}] ={v:.2f}')
    plt.tight_layout()
    plt.colorbar(pmap, ax=axs, fraction=0.01)
    plt.show()
