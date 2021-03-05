function [xtemp, k, time, FF,  Error, a]=FISTA_exp(x,para,ProX,GradF,FvalF,FG_val)
% This function uses IFBS with t_k = exp((k-1).^r) to solve l1-regularized logistic regression
% problem and Lasso problem. The decription of the algorithm can be found in our paper:

% H.W. Liu, T. Wang, Z.X. Liu:
% "Convergence Rate of Inertial Forward-Backward Splitting Algorithms 
%  Based on the Local Error Bound Condition"

% Inputs:
%   1. x: initial point
%   2. para: parameters
%   3. ProX: proximity operator
%   4. GradF,FvalF,FG_val: calculation of gradient value and function value
%      of f(x)=0.5*||A*x-b||^2

% Outputs:
%   1. xtemp: solution
%   2. k: the number of iterations
%   3. time: cpu time
%   4. FF: the array of function values of each iterate
%   5. Error: the array of gradient values of function F(x)=f(x)+g(x) of each iterate
%   6. a: step length

    format long
    global ebs
    maxiter=para.maxiter;
    lamda=para.lamda;
    Index=para.Index;
    a=para.a;
    FF=zeros(maxiter,1);
    Error=FF; 
    y=x;
    xold=x;
    r=para.r; % t_k = exp((k-1).^r)
    
    tic
    grad_y=GradF(y);
    u=lamda*a;
    z=y-a*grad_y;
    xtemp=ProX(u,z,Index);
    diff_val=xtemp-y;   
    [F, grad_xtemp]=FG_val(xtemp);
    F=lamda*sum(abs(xtemp))+F; 
    diff_grad=grad_xtemp-grad_y; 
    error=norm(diff_val-a*(diff_grad))/a;
    k=1;   
    FF(k)=F;Error(k)=error;
 
    while error>ebs        
        tmp=exp((k-1).^r-k.^r)-exp(-(k.^r));
        y=xtemp+tmp*(xtemp-xold);
        xold=xtemp;
        grad_y=GradF(y);
        u=lamda*a;
        z=y-a*grad_y;
        xtemp=ProX(u,z,Index);
        
        diff_val=xtemp-y;           
        [F, grad_xtemp]=FG_val(xtemp);
        F=lamda*sum(abs(xtemp))+F;
        diff_grad=grad_xtemp-grad_y;        
        error=norm(diff_val-a*(diff_grad))/a;
        k=k+1;      
        FF(k)=F;Error(k)=error;
        if k>maxiter
            break;
        end
    end
    FF(k+1:maxiter)=[];
    Error(k+1:maxiter)=[];
    time=toc;
end
