function func_val = fval_least_squa(x,para)
A=para.A;
b=para.b;

w=A*x-b;
func_val=0.5*(w')*(w);
end