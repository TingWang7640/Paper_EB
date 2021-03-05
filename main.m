clc
clear
close all
addpath data_FISTA
addpath tool_FISTA

%%
%%
global ebs 
ebs=1.e-8;
%% ① Sparse Logistic Regression 
%%
%     Index=1;
%     maxiter=50000;
%     lamda=1.e-2; %正则化参数
% 
% %load('sonar_test.mat'); 
% %load('w4a_h.mat'); 
% %load('w4a_l.mat'); 
% load('a9a_test.mat');
%     para.l=l;
%     para.h=h;
%     [n,m] = size(h);
%     para.m = m;
%     para.n = n;
%   
%     w=diag(l);
%     K=w*h;
%     K=K.'*K;
%     L=4*n/norm(K); % L=1/L(f)
%     %L = 4*n/eigs(h*h',1,'LM');
% 
%     GradF=@(x) grad_logistic(x, para); 
%     FvalF=@(x) fval_logistic(x, para);
%     FG_val=@(x) fg_val_logistic(x,para);     
%% ② Lasso   f(x)=0.5*||A*x-b||^2,g(x)=lamda*||x||_1
%%   
    Index=2;
    lamda=1;
    maxiter=100000;
    n=500;m=5000;
    A=randn(n,m);
    xstar= ones(m,1);
    nnzNum=50;
    I=randperm(m); xstar(I(1:m-nnzNum))=0;
    b=A*xstar+randn(n,1)*0.5;
    para.A=A;
    para.b=b;   
    L=1/norm(A*A');  % L=1/L(f)

    FvalF=@(x) fval_least_squa(x,para);  
    GradF=@(x) grad_least_squa(x,para);
    FG_val=@(x) fg_val_least_squa(x,para);    
%% Parameter
%%    
    para.m = m;
    para.n = n; 
    para.maxiter=maxiter;
    para.Index=Index;
    para.lamda=lamda;
    para.a=0.98*L; % step length

    x0 = zeros(m,1); % Initial point   
    ProX=@(u,z,Index) proxf(u,z,Index,para);
%%
%%
% FISTA
    [x1,k1,t1, F1, SUDI1, a1]=FISTA(x0,para,ProX,GradF,FvalF,FG_val);   
% FISTA_CD4
    para.ss=4; %t_k = (k+ss-1)/ss
    [x2,k2,t2, F2, SUDI2, a2]=FISTA_CD4(x0,para,ProX,GradF,FvalF,FG_val);
% FISTA_pow    
    para.ss=4;para.r=8;  %t_k = ((k.^r)+ss-1)/ss
    [x3,k3,t3, F3, SUDI3, a3]=FISTA_pow(x0,para,ProX,GradF,FvalF,FG_val);
    %
    para.ss=0.5;para.r=0.5;
    [x4,k4,t4, F4, SUDI4, a4]=FISTA_pow(x0,para,ProX,GradF,FvalF,FG_val); 
% FISTA_exp
	para.r=0.5; % t_k = exp((k-1).^r)
    [x5,k5,t5, F5, SUDI5, a5]=FISTA_exp(x0,para,ProX,GradF,FvalF,FG_val);
%    
    Fmin=min([min(F1),min(F2),min(F3),min(F4),min(F5)]);
    F1=F1-Fmin;
    F2=F2-Fmin;
    F3=F3-Fmin;
    F4=F4-Fmin;
    F5=F5-Fmin;

figure
ax1 = subplot(1,1,1);
P1  = semilogy((0:(k1-1)),SUDI1,'b','LineWidth',1.1); hold on
P2  = semilogy((0:(k2-1)),SUDI2,'k','LineWidth',1.1); hold on
P3  = semilogy((0:(k3-1)),SUDI3,'m','LineWidth',1.1); hold on
P4  = semilogy((0:(k4-1)),SUDI4,'c','LineWidth',1.1); hold on
P5  = semilogy((0:(k5-1)),SUDI5,'r','LineWidth',1.1); hold on

title(ax1,'n=500, m=5000, s=50')
legend('FISTA','FISTA\_CD','FISTA\_pow(8)','FISTA\_pow(0.5)','FISTA\_exp')
xlabel(ax1,'Iteration','FontSize',10)
ylabel(ax1,'${\left\| {{\psi _k}} \right\|}$','interpreter','latex','FontSize',10)

figure
ax1 = subplot(1,1,1);
T1  = semilogy((0:(k1-1)),F1,'b','LineWidth',1.1); hold on
T2  = semilogy((0:(k2-1)),F2,'k','LineWidth',1.1); hold on
T3  = semilogy((0:(k3-1)),F3,'m','LineWidth',1.1); hold on
T4  = semilogy((0:(k4-1)),F4,'c','LineWidth',1.1); hold on
T5  = semilogy((0:(k5-1)),F5,'r','LineWidth',1.1); hold on

title(ax1,'n=500, m=5000, s=50')
legend('FISTA','FISTA\_CD','FISTA\_pow(8)','FISTA\_pow(0.5)','FISTA\_exp')
xlabel(ax1,'Iteration','FontSize',10)
ylabel(ax1,'${F\left( {{x_k}} \right) - {F^ * }}$','interpreter','latex','FontSize',10)

