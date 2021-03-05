function [func_val, grad_x] = fg_val_logistic(x,para)
    n=para.n;
    h=para.h;
    l=para.l;
    p=zeros(n,1);
 
    w=h*x;
    w=l.*w;
    I=logical(w>0);
    J=~I;

    p(I)=log(1+exp(-w(I)));
    p(J)=log(1+exp(w(J)))-w(J);
    func_val=mean(p);
    

    tmp=exp(-w(I));
    p(I)=tmp.*(-l(I))./(1+tmp);
    p(J)=(-l(J))./(1+exp(w(J)));
    grad_x=(h'*p)/n;


end