import numpy as np

X = [1.2,2.6]
Y = [1.0,1.1]

def f(w,b,x):
    return 1.0/(1.0 + np.exp(-(w*x + b)))

def error(w,b):
    err = 0
    for x,y in zip(X,Y):
        fx = f(w,b,x)
        err += 0.5 * (fx - y)**2
    return err
    
def grad_b(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y) * fx * (1-fx)

def grad_w(w,b,x,y):
    fx = f(w,b,x)
    return (fx-y) * fx * (1-fx) * x

def do_momentum_gradient_descent():
    w,b,eta,max_epochs = -4,-4,1.0,100
    prev_v_w, prev_v_b, gamma = 0, 0, 0.9
    for i in range(max_epochs):
        dw,db = 0,0
        for x,y in zip(X,Y):
            
            dw += grad_w(w,b,x,y)
            db += grad_b(w,b,x,y) 
            
        v_w = gamma * prev_v_w + eta * dw
        v_b = gamma * prev_v_b + eta * db
            
        w = w - v_w
        b = b - v_b
        prev_v_w = v_w
        prev_v_b = v_b
        err = error(w,b)
        print(err)
            
            
do_momentum_gradient_descent()