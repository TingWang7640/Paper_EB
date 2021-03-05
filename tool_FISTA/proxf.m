function prox_val=proxf(u,z,Index,para)

    n=length(z);
    prox_val=zeros(n,1);
    if Index==1 || Index==2
    
        I=logical((u>=z)&( u<-z));
        prox_val(I)=z(I)+u;
        I=logical(u<z);
        prox_val(I)=z(I)-u;
    
    end
    

    
end
