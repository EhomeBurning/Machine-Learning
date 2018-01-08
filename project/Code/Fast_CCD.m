function dec=Fast_ccd(y,g,cut,alp,alpha)

M=length(y);
%% Compute Test Statistics 
Ts=0;
for i=1:M
%   grid_n=find(grid<=y_i,1,'last');
    Ts=Ts+(pdf(g,y(i))<cut);
end 


%% Compute the Decision Parameters 
p_v=binocdf(Ts,M,alp,'upper') ;



% 
if p_v>=alpha
    dec=0;
else 
    dec=1;
end 

end 