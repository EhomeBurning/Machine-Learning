clear all
close all
clc


%% Specify Parameters
alpha=0.1;
ccd_acc=0;

%% Generate Nominal Data
load http.mat

%% Set the Grid
grid_x=-9:0.1:11;
grid_y=-1.5:0.1:11;

X=X-mean(X,1);
N=length(X);
X1=X;
idx=randsample(N,20000);
X=X(idx,:);
y1=y(idx,:);
% y1=y;
[~,~,V]=svd(X,'econ');
v=V(:,1:2);
X=X*v;
X1=X1*v;

x_train=X(y1==0,:);
N=length(x_train);
x_train=x_train(randsample(N,1000),:);


x_test=X1;

y_test=y;

N=length(x_train);
M=length(x_test);
n=floor(N/2);

idx=randsample(N,N);
x1=x_train(idx(1:n),:);
x2=x_train(idx(n+1:end),:);

[g1,g2,cut,alp]=CCD_con_2D(x1,x2,alpha,grid_x,grid_y);
dec_ccd=pdf(g1,x_test(:,1)).*pdf(g2,x_test(:,2))<cut;

%% Output Values  
FAR=sum(dec_ccd(y_test==0))/sum(y_test==0); %False Alert Rate
CRR=sum(dec_ccd(y_test==1))/sum(y_test==1); %Correct Rejection Rate
% acc=sum(dec_ccd==y_test)/length(y_test); %Total Prediction Accuracy 
 
summary=[FAR,CRR]
% fprintf('Correct Rejection Rate: ',num2str(CRR),'\n','False Alarm Rate: ',num2str(FAR),'\n','Significant Level Set',num2str(alpha),'\n');