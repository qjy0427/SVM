function svm = svmTrain(X,Y,C)

options = optimset;    
options.LargeScale = 'off';
options.Display = 'off';    

% Quadratic programming
n = length(Y);
H = (Y'*Y).*kernel(X,X);    
f = -ones(n,1); 
A = [];
b = [];
Aeq = Y;
beq = 0;
lb = zeros(n,1); 
ub = C*ones(n,1);
a0 = zeros(n,1);
a = quadprog(H,f,A,b,Aeq,beq,lb,ub,a0,options);
epsilon = 1e-6;  
sv_label = find(abs(a)>epsilon);
svm.a = a(sv_label);
svm.Xsv = X(:,sv_label);
svm.Ysv = Y(sv_label);

end