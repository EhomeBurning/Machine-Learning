clear all
close all
clc


%% Specify Parameters
alpha=0.01;
ccd_acc=0;

%% Generate Nominal Data
load http.mat

%% Set the Grid
grid_x=-3:0.1:10;
grid_y=-3:0.1:13;
grid_z=-3:0.1:18;

%% Date
N=length(X);
X1=X;

%% Use 10000 for training
idx=randsample(N,10000);
X=X(idx,:);
% X1=X1-mean(X);
% X=X-mean(X);
y1=y(idx,:);


%% Among the training Data only take 1000 of them as nominal set
x_train=X(y1==0,:);
% N=length(x_train);
% x_train=x_train(randsample(N,1000),:);
%% Use the whole Set as Test Data
x_test=X1;
y_test=y;

%% Training Data Prameters 
N=length(x_train);
M=length(x_test);
n=floor(N/2);

%% Sample Splitting 
idx=randsample(N,N);
x1=x_train(idx(1:n),:);
x2=x_train(idx(n+1:end),:);

% Construct the Decision Rule Based on Conformal Prediction
[g1,g2,g3,cut,alp]=CCD_con_3D(x1,x2,alpha,grid_x,grid_y,grid_z);
eta=pdf(g1,x_test(:,1)).*pdf(g2,x_test(:,2)).*pdf(g3,x_test(:,3));
dec_ccd=eta<cut;

%% Output Values  
FAR=sum(dec_ccd(y_test==0))/sum(y_test==0); %False Alert Rate
CRR=sum(dec_ccd(y_test==1))/sum(y_test==1); %Correct Rejection Rate
% acc=sum(dec_ccd==y_test)/length(y_test); %Total Prediction Accuracy 
 
summary=[FAR,CRR]
% fprintf('Correct Rejection Rate: ',num2str(CRR),'\n','False Alarm Rate: ',num2str(FAR),'\n','Significant Level Set',num2str(alpha),'\n');