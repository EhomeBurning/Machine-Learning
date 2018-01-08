clear all
close all
clc


%% Specify Parameters
alpha=0.15;
ccd_acc=0;

%% Generate Nominal Data
load http.mat

% coeff = pca(X);
% [~,~,V]=svd(X-mean(X,1),'econ');
% X=X-mean(X,1);
% [~,~,V]=svd(X,'econ');
% v=V(:,1);
% X1= X * coeff(:,1);
X=X-mean(X,1);
N=length(X);
X1=X;
idx=randsample(N,20000);
X=X(idx,:);
y1=y(idx,:);
% y1=y;
[~,~,V]=svd(X,'econ');
v=V(:,1);
X=X*v;
X1=X1*v;

% X_train=X(y==0);
x_train=X(y1==0);
N=length(x_train);
% X_train=X_train-mean(X_train,2);
% x_train=X(y==0);
x_train=x_train(randsample(N,1000));
% X_train=X_train(randsample(N,1000));

x_test=X1;
% X_test=X(y==1);
% X_test=X_test-mean(X_test,2);
y_test=y;

% [U S V]=svd(X_train,'econ');
% u=U(:,1);
% v=V(:,1);
% s=S(1,1);
% 
% x_train=X_train*v';
% x_test=X_test*v';

N=length(x_train);
M=length(x_test);
n=floor(N/2);

idx=randsample(N,N);
x1=x_train(idx(1:n));
x2=x_train(idx(n+1:end));

[g,cut,alp]=CCD_con(x1,x2,alpha);
dec_ccd=pdf(g,x_test)<cut;
acc=sum(dec_ccd==y_test);

% for i=1:M
% 
% 
% %% Testing Different Method
% % CCD
% % dec_ccd=Fast_CCD(x_test(i),g,cut,alp,alpha);
% dec_ccd=pdf(g,y(i))<cut
% 
% %Summarizing Statistics
% ccd_acc=ccd_acc+(dec_ccd==y_test(i));
% end 

