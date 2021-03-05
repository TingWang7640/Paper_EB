function grad_x = grad_least_squa(x,para)
    A=para.A;
    b=para.b;

    w=A*x-b;
    grad_x = A'*w;
end