function grad_x = grad_logistic(x,para)

    n=para.n;
    h=para.h;
    l=para.l;
    p=zeros(n,1);

    w=h*x;
    w=l.*w;
    I=logical(w>0);
    J=~I;
    tmp=exp(-w(I));
    p(I)=tmp.*(-l(I))./(1+tmp);
    p(J)=(-l(J))./(1+exp(w(J)));
    grad_x=(h'*p)/n;
    
end