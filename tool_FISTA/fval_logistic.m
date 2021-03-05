function func_val = fval_logistic(x, para)

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
end