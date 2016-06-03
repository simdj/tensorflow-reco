clear all;

data = zeros(10,30);

romantic_lover = 1:5;
horror_lover = 6:10;
romantic_movie=1:10;
horror_movie=11:20;
common_movie = 21:30;

data(romantic_lover,romantic_movie)=10*ones(5,10);
data(horror_lover,romantic_movie)=1*ones(5,10);

data(romantic_lover,horror_movie)=1*ones(5,10);
data(horror_lover,horror_movie)=10*ones(5,10);

data(romantic_lover,common_movie)=5*ones(5,10);
data(horror_lover,common_movie)=5*ones(5,10);

data=abs(awgn(data,1));




%% row_rank SVD
[U,S,V]=svd(data);
U=U(:,1:2);
S=S(1:2,1:2);
V=V(:,1:2);
estimated_data = U*S*V';
error_naive = sum(sum((data-estimated_data).^2));

%% selection bias
% 1. the lower rating, the less selected
data_bias(romantic_lover,romantic_movie)=masking_data(data(romantic_lover,romantic_movie),0.8);
data_bias(romantic_lover,horror_movie)=masking_data(data(romantic_lover,horror_movie),0.2);
data_bias(horror_lover,romantic_movie)=masking_data(data(horror_lover,romantic_movie),0.8);
data_bias(horror_lover,horror_movie)=masking_data(data(horror_lover,horror_movie),0.2);
data_bias(romantic_lover,common_movie)=masking_data(data(romantic_lover,common_movie),0.6);
data_bias(horror_lover,common_movie)=masking_data(data(horror_lover,common_movie),0.6);

[U,S,V]=svd(data_bias);

U=U(:,1:5);
S=S(1:5,1:5);
V=V(:,1:5);
estimated_data = U*S*V';
% bias data naive test
observed_error_naive = sum(sum((data_bias(data_bias>0)-estimated_data(data_bias>0)).^2));
% total data
total_error_naive = sum(sum((data-estimated_data).^2));


%% error function unbiased version

%% estimate propensity


O=double(data_bias>0);
[U,S,V]=svd(O);
U=U(:,1:5);
S=S(1:5,1:5);
V=V(:,1:5);
O_estimated = U*S*V';