function [func_val, grad_x] = fg_val_least_squa(x,para)
A=para.A;
b=para.b;


w=A*x-b;
func_val=0.5*(w')*(w);
grad_x = A'*w;
end