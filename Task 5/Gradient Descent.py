import numpy as np

X = [1.2,2.6]
Y = [1.0,1.1]

def f(w,b,x): #sigmoid
    return 1.0/(1.0 + np.exp(-(w*x + b)))

def error(w,b): #error/loss function
    err = 0.0
    for x,y in zip(X,Y):
       fx = f(w,b,x)
       err+= 0.5 * (fx - y)**2 
    return err

def grad_b(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y) * fx * (1-fx)

def grad_w(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y) * fx * (1-fx) * x

def do_gradient_descent():
    w,b,eta,max_epochs = -4,-4,1.0,100
    for i in range(max_epochs):
        dw, db=0,0
        for x,y in zip(X,Y):
            dw += grad_w(w,b,x,y)
            db += grad_b(w,b,x,y)
        w = w - eta * dw
        b = b - eta * db
        
        err = error(w,b)
        print(err)
        
do_gradient_descent()